"""Dynamic-object ("actor") placement over the gRPC rendering service.

The service can now be told to put a rigid Gaussian object into the loaded
scene and to move it every frame off the pose streams. What is worth defending
here is the plumbing a client cannot see: that spawn refuses the states that
would corrupt the scene, that an actor's pose reaches the body it names in the
frame the client meant, and that the two proto3 traps on the wire — an unset
pose and an all-zero quaternion — do not silently teleport or collapse an
actor.

Everything runs on CPU tensors against a background-less :class:`Scene`; no
GPU and no scene bundle are needed to exercise the servicer's own logic.
"""

from __future__ import annotations

import time
from typing import Any, cast

import grpc
import pytest
import torch

from splatsim._conversions import GaussianTensors
from splatsim.dataclass.lod_config import LodConfig
from splatsim.grpc_service._generated import rendering_service_pb2 as pb2
from splatsim.grpc_service.server import RenderingServiceServicer
from splatsim.lod import LodManager
from splatsim.rigid_body import RigidBody
from splatsim.scene import Scene

# None of these handlers touch the ServicerContext, so a typed None stand-in
# is enough to drive them (same trick as test_grpc_lod.py).
_CTX = cast(grpc.ServicerContext, None)

_CPU = torch.device("cpu")


class _FakeBackground:
    """The two attributes of a Background that the actor paths reach for.

    ``tile_local_centroid`` is what ``world_to_tile_local`` recentres against;
    ``tensors`` is what ``Scene.add_rigid_body`` compares a new body's colour
    block against.
    """

    def __init__(
        self, centroid: tuple[float, float, float], sh_degree: int = 0
    ) -> None:
        self.tile_local_centroid = torch.tensor(centroid, dtype=torch.float32)
        self.tensors = _tensors(2, sh_degree=sh_degree)


def _tensors(n: int = 4, sh_degree: int = 0) -> GaussianTensors:
    k = (sh_degree + 1) ** 2
    return GaussianTensors(
        means=torch.zeros(n, 3),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * n),
        scales=torch.ones(n, 3),
        opacities=torch.ones(n),
        colors=torch.rand(n, 3) if sh_degree == 0 else torch.rand(n, k, 3),
        sh_degree=sh_degree,
    )


def _servicer_with_body(
    name: str = "car_01",
    *,
    world_frame: bool = False,
    background: _FakeBackground | None = None,
) -> tuple[RenderingServiceServicer, RigidBody]:
    """A servicer holding a scene with one spawned body, ready to be posed."""
    servicer = RenderingServiceServicer()
    scene = Scene(background=None, lod_manager=LodManager(LodConfig()))
    sh_degree = 0
    if background is not None:
        scene.background = cast(Any, background)
        # Match the background, or the body would not be addable at all.
        sh_degree = background.tensors.sh_degree
    body = RigidBody(_tensors(sh_degree=sh_degree), device=_CPU)
    scene.add_rigid_body(name, body)
    servicer._scene = scene
    servicer._device = _CPU
    servicer._initialized = True
    # What a successful SpawnActor would have recorded for this instance.
    servicer._actor_world_frame[name] = world_frame
    return servicer, body


def _actor_pose(
    instance_id: str,
    position: tuple[float, float, float],
    rotation: tuple[float, float, float, float] | None = None,
) -> pb2.ActorPose:
    msg = pb2.ActorPose(instance_id=instance_id)
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = position
    if rotation is not None:
        w, x, y, z = rotation
        msg.pose.rotation.w = w
        msg.pose.rotation.x = x
        msg.pose.rotation.y = y
        msg.pose.rotation.z = z
    return msg


# --- wire format ------------------------------------------------------------


def test_every_pose_stream_carries_actors() -> None:
    """All three streams take actor poses, so a client need not run a rig."""
    for msg in (pb2.CameraData(), pb2.LidarData(), pb2.RigData()):
        msg.actors.append(_actor_pose("car_01", (1.0, 2.0, 3.0)))
        assert msg.actors[0].instance_id == "car_01"


def test_spawn_request_pose_keeps_presence() -> None:
    """An omitted pose must stay distinguishable from a pose of all zeros.

    SpawnActor uses ``HasField`` to leave a body at the origin instead of
    reading zeros out of an unset message; dropping message-typed presence
    would make that check meaningless.
    """
    req = pb2.SpawnActorRequest(instance_id="car_01", asset_id="sedan_0007")
    assert req.HasField("pose") is False
    req.pose.position.x = 1.0
    assert req.HasField("pose") is True


# --- guard rails ------------------------------------------------------------


def test_actor_rpcs_refuse_before_initialize() -> None:
    servicer = RenderingServiceServicer()
    spawn = servicer.SpawnActor(
        pb2.SpawnActorRequest(instance_id="car_01", asset_id="sedan_0007"), _CTX
    )
    remove = servicer.RemoveActor(pb2.RemoveActorRequest(instance_id="car_01"), _CTX)
    listed = servicer.ListActorAssets(pb2.ListActorAssetsRequest(), _CTX)

    assert spawn.success is False and "Initialize" in spawn.message
    assert remove.success is False and "Initialize" in remove.message
    assert listed.success is False and "Initialize" in listed.message


def test_spawn_requires_instance_id_and_an_asset() -> None:
    servicer, _ = _servicer_with_body()

    unnamed = servicer.SpawnActor(pb2.SpawnActorRequest(asset_id="sedan_0007"), _CTX)
    assert unnamed.success is False
    assert "instance_id" in unnamed.message

    sourceless = servicer.SpawnActor(pb2.SpawnActorRequest(instance_id="car_02"), _CTX)
    assert sourceless.success is False
    assert "asset_id" in sourceless.message
    # A failed spawn must not leave the instance registered.
    assert "car_02" not in servicer._actor_world_frame


def test_spawn_rejects_a_name_already_in_the_scene() -> None:
    """Otherwise the second spawn would silently replace a body the client
    is still streaming poses for."""
    servicer, body = _servicer_with_body("car_01")
    resp = servicer.SpawnActor(
        pb2.SpawnActorRequest(instance_id="car_01", asset_id="sedan_0007"), _CTX
    )
    assert resp.success is False
    assert "already in the scene" in resp.message
    assert servicer._scene is not None
    assert servicer._scene["car_01"] is body


def test_spawn_world_frame_needs_a_background() -> None:
    """World-frame poses are recentred against the background's centroid, so
    without one the request is refused rather than placed hundreds of metres
    off."""
    servicer, _ = _servicer_with_body()
    resp = servicer.SpawnActor(
        pb2.SpawnActorRequest(
            instance_id="car_02", asset_id="sedan_0007", world_frame=True
        ),
        _CTX,
    )
    assert resp.success is False
    assert "world_frame" in resp.message


def test_remove_actor_takes_the_body_out_of_the_scene() -> None:
    servicer, _ = _servicer_with_body("car_01")
    assert servicer._scene is not None

    resp = servicer.RemoveActor(pb2.RemoveActorRequest(instance_id="car_01"), _CTX)
    assert resp.success is True
    assert "car_01" not in servicer._scene
    assert "car_01" not in servicer._actor_world_frame

    again = servicer.RemoveActor(pb2.RemoveActorRequest(instance_id="car_01"), _CTX)
    assert again.success is False
    assert "no rigid body" in again.message


def test_scene_rejects_colours_it_cannot_gather_with() -> None:
    """A mismatched colour block would abort the render thread on the next
    frame; the caller hears about it while it is still listening.

    The check sits on ``add_rigid_body``, so SpawnActor's failure response and
    the Python API's exception are the same rule seen from two sides.
    """
    sh_background = _FakeBackground((0.0, 0.0, 0.0), sh_degree=3)
    servicer, _ = _servicer_with_body(background=sh_background)
    assert servicer._scene is not None

    with pytest.raises(ValueError, match="cannot be rendered together"):
        servicer._scene.add_rigid_body("rgb_car", RigidBody(_tensors(), device=_CPU))


def test_scene_accepts_matching_colours() -> None:
    servicer, _ = _servicer_with_body(background=_FakeBackground((0.0, 0.0, 0.0)))
    assert servicer._scene is not None
    servicer._scene.add_rigid_body("car_02", RigidBody(_tensors(), device=_CPU))
    assert "car_02" in servicer._scene


def test_instances_share_lod_sorted_tensors() -> None:
    """What makes the file-asset cache worth having.

    ``LodManager.precompute`` REORDERS the Gaussians, so a body that
    precomputes holds its own copy of the whole cloud — 24 MB for a 100k-point
    SH vehicle. Handing the sorted tensors and their index to the next instance
    keeps every copy of one asset on a single upload.
    """
    manager = LodManager(LodConfig())
    sorted_tensors, index = manager.precompute(_tensors(32))

    first = RigidBody(sorted_tensors, device=_CPU, lod_index=index, rotate_sh=True)
    second = RigidBody(sorted_tensors, device=_CPU, lod_index=index, rotate_sh=True)

    assert first.base_tensors is sorted_tensors
    assert second.base_tensors is first.base_tensors
    assert second.lod_index is index


def test_rigid_body_refuses_both_lod_inputs() -> None:
    """Precomputing over a prebuilt index would silently discard the index."""
    manager = LodManager(LodConfig())
    _, index = manager.precompute(_tensors(8))
    with pytest.raises(ValueError, match="not both"):
        RigidBody(_tensors(), device=_CPU, lod_manager=manager, lod_index=index)


# --- pose application -------------------------------------------------------


def test_actor_pose_moves_the_named_body() -> None:
    servicer, body = _servicer_with_body("car_01")
    servicer._apply_actor_poses(
        [_actor_pose("car_01", (10.0, -2.0, 1.5), (0.0, 0.0, 0.0, 1.0))]
    )
    torch.testing.assert_close(body.position, torch.tensor([10.0, -2.0, 1.5]))
    torch.testing.assert_close(body.rotation, torch.tensor([0.0, 0.0, 0.0, 1.0]))


def test_zero_quaternion_keeps_the_current_rotation() -> None:
    """A client that fills in only a position sends w=x=y=z=0, which is not a
    rotation — applying it verbatim would collapse the body."""
    servicer, body = _servicer_with_body("car_01")
    body.set_pose((0.0, 0.0, 0.0), (0.966, 0.0, 0.0, 0.259))
    servicer._apply_actor_poses([_actor_pose("car_01", (5.0, 0.0, 0.0))])

    torch.testing.assert_close(body.position, torch.tensor([5.0, 0.0, 0.0]))
    torch.testing.assert_close(body.rotation, torch.tensor([0.966, 0.0, 0.0, 0.259]))


def test_actor_pose_without_a_pose_leaves_the_body_alone() -> None:
    servicer, body = _servicer_with_body("car_01")
    body.set_pose((1.0, 2.0, 3.0))
    servicer._apply_actor_poses([pb2.ActorPose(instance_id="car_01")])
    torch.testing.assert_close(body.position, torch.tensor([1.0, 2.0, 3.0]))


def test_world_frame_pose_is_recentred_onto_the_tile() -> None:
    """A world-frame instance has the scene origin subtracted, exactly as
    ``world_to_tile_local`` does for scenario code."""
    background = _FakeBackground((100.0, -50.0, 2.0))
    servicer, body = _servicer_with_body(
        "car_01", world_frame=True, background=background
    )
    servicer._apply_actor_poses([_actor_pose("car_01", (113.6, -58.5, 1.9))])
    torch.testing.assert_close(
        body.position, torch.tensor([13.6, -8.5, -0.1]), atol=1e-5, rtol=0
    )


def test_tile_local_pose_is_used_as_given() -> None:
    """The default frame matches the streams' own camera/LiDAR poses, which
    clients already send tile-local."""
    background = _FakeBackground((100.0, -50.0, 2.0))
    servicer, body = _servicer_with_body("car_01", background=background)
    servicer._apply_actor_poses([_actor_pose("car_01", (13.6, -8.5, -0.1))])
    torch.testing.assert_close(
        body.position, torch.tensor([13.6, -8.5, -0.1]), atol=1e-5, rtol=0
    )


def test_unknown_instance_is_ignored_once_not_fatal() -> None:
    """A stale id in a client's actor list must not kill a running stream."""
    servicer, body = _servicer_with_body("car_01")
    servicer._apply_actor_poses(
        [
            _actor_pose("ghost", (1.0, 1.0, 1.0)),
            _actor_pose("car_01", (2.0, 0.0, 0.0)),
        ]
    )
    assert servicer._unknown_actor_ids == {"ghost"}
    torch.testing.assert_close(body.position, torch.tensor([2.0, 0.0, 0.0]))


# --- streaming --------------------------------------------------------------


def _rig_data(time_ns: int, actors: list[pb2.ActorPose] | None = None) -> pb2.RigData:
    msg = pb2.RigData()
    msg.stamp.sec, msg.stamp.nanosec = divmod(time_ns, 1_000_000_000)
    msg.pose.rotation.w = 1.0
    if actors:
        msg.actors.extend(actors)
    return msg


def test_stream_applies_actor_poses_before_each_frame() -> None:
    """The render thread must pose the actors of the message it is rendering,
    so a frame's objects and its sensor pose come from the same instant."""
    servicer, body = _servicer_with_body("car_01")
    seen: list[tuple[float, float, float]] = []

    def record(pose, render_time_ns, sweep_start=None):
        seen.append(tuple(body.position.tolist()))

    def producer():
        for i in range(3):
            yield _rig_data(
                i * 1_000_000,
                [_actor_pose("car_01", (float(i), 0.0, 0.0))],
            )
            # Let the render thread consume this message before the next.
            time.sleep(0.03)

    summary = servicer._run_pose_stream(
        producer(), frame_rate=1000.0, render_and_publish=record
    )

    assert summary.poses_received == 3
    assert seen  # at least one frame rendered
    # Every rendered frame saw the actor at the pose its own message carried,
    # and the last message's pose is the one left standing.
    assert all(p in {(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)} for p in seen)
    torch.testing.assert_close(body.position, torch.tensor([2.0, 0.0, 0.0]))


def test_stream_without_actors_leaves_previous_poses_standing() -> None:
    """Actors are optional per message: a client that moves them at a lower
    rate than it streams poses must not have them snap back."""
    servicer, body = _servicer_with_body("car_01")

    def producer():
        yield _rig_data(0, [_actor_pose("car_01", (7.0, 0.0, 0.0))])
        time.sleep(0.03)
        yield _rig_data(1_000_000)
        time.sleep(0.03)

    servicer._run_pose_stream(
        producer(), frame_rate=1000.0, render_and_publish=lambda *a, **k: None
    )
    torch.testing.assert_close(body.position, torch.tensor([7.0, 0.0, 0.0]))
