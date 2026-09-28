// Fused LiDAR cull kernel: computes a single boolean keep mask combining
// the shell/frustum test (min_range, max_range), the elevation-FOV test
// against a spherical band around the sensor's up-axis, and (optionally) an
// azimuth-wedge test for sector rendering.
//
// All arithmetic is done in fp32; a single pass over means (Nx3) and
// scales (Nx3) reads 24 bytes per gaussian and writes 1 byte to `keep`,
// dwarfing the ~130 bytes/gaussian traffic of the PyTorch chain.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <type_traits>

namespace {

template <bool USE_ELEV, bool USE_AZIM>
__global__ void lidar_cull_kernel(
    const float* __restrict__ means,       // (N, 3) row-major
    const float* __restrict__ scales,      // (N, 3) row-major
    const float* __restrict__ sensor_pos,  // (3,) device pointer, read once per block
    const float* __restrict__ up_world,    // (3,) device pointer, ignored when !USE_ELEV
    const float* __restrict__ fwd_world,   // (3,) sensor +x in world, ignored when !USE_AZIM
    const float* __restrict__ left_world,  // (3,) sensor +y in world, ignored when !USE_AZIM
    const float min_range,
    const float max_range,        // <0 means "no upper bound"
    const float cull_scale_sigmas,
    const float sin_min,
    const float cos_min,
    const float sin_max,
    const float cos_max,
    const float az_center,        // wedge centre azimuth [rad], sensor frame
    const float az_halfwidth,     // wedge half-width [rad]
    const float az_pad,           // fixed extra angular slack [rad]
    const int32_t N,
    bool* __restrict__ keep) {
  // Cache the sensor pose in shared memory so all N threads amortize a
  // single global fetch. sensor_pos and up_world are tiny (3 floats each)
  // and passing them via device pointers avoids the launch-side sync we
  // used to pay to read them as scalar kernel args.
  __shared__ float s_sensor[3];
  __shared__ float s_up[3];
  __shared__ float s_fwd[3];
  __shared__ float s_left[3];
  if (threadIdx.x < 3) {
    s_sensor[threadIdx.x] = sensor_pos[threadIdx.x];
    if (USE_ELEV) {
      s_up[threadIdx.x] = up_world[threadIdx.x];
    }
    if (USE_AZIM) {
      s_fwd[threadIdx.x] = fwd_world[threadIdx.x];
      s_left[threadIdx.x] = left_world[threadIdx.x];
    }
  }
  __syncthreads();

  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= N) {
    return;
  }

  const int64_t idx3 = i * 3;
  // Load means + scales as (N, 3) row-major. Not 16B-aligned, so use
  // scalar loads and rely on the compiler to coalesce. On sm_89 this is
  // essentially the same throughput as float4.
  const float mx = means[idx3 + 0];
  const float my = means[idx3 + 1];
  const float mz = means[idx3 + 2];

  const float sx = scales[idx3 + 0];
  const float sy = scales[idx3 + 1];
  const float sz = scales[idx3 + 2];

  // delta = means - sensor_pos
  const float dx = mx - s_sensor[0];
  const float dy = my - s_sensor[1];
  const float dz = mz - s_sensor[2];

  // dist = |delta|
  const float dist = sqrtf(dx * dx + dy * dy + dz * dz);

  // max_scale (nan → 0).
  // Match torch.nan_to_num(scales.amax(...), nan=0.0): if *any* scale
  // component is NaN, amax returns NaN and we then map it to zero.
  // CUDA's fmaxf skips NaN inputs (per IEEE 754), so we can't just
  // rely on it — check explicitly first.
  float ms;
  if (isnan(sx) || isnan(sy) || isnan(sz)) {
    ms = 0.0f;
  } else {
    ms = fmaxf(fmaxf(sx, sy), sz);
  }
  const float margin = cull_scale_sigmas * ms;

  // Shell test: overlap [dist - margin, dist + margin] with [min_range, max_range]
  bool k = (dist + margin) >= min_range;
  if (max_range >= 0.0f) {
    k = k && ((dist - margin) <= max_range);
  }

  // Elevation-FOV test (linear-in-margin bound):
  //   z_s = dot(delta, up_world)
  //   sin(elev) = z_s / dist  (guarded: dist>0 handled below)
  //   keep if
  //     z_s + margin·cos(elev_min) >= dist·sin(elev_min)   (above lower plane)
  //     z_s - margin·cos(elev_max) <= dist·sin(elev_max)   (below upper plane)
  if (USE_ELEV && k) {
    const float z_s = dx * s_up[0] + dy * s_up[1] + dz * s_up[2];
    k = k && ((z_s + margin * cos_min) >= (dist * sin_min));
    k = k && ((z_s - margin * cos_max) <= (dist * sin_max));
  }

  // Azimuth-wedge test (sector rendering): keep iff the Gaussian's angular
  // extent can overlap the wedge. The angular margin divides by the
  // HORIZONTAL distance, not the slant range: azimuth is atan2(y, x), so a
  // Gaussian near the elevation extremes subtends a wider azimuth arc than
  // its slant-range extent suggests (by 1/cos(elev)); dividing by the slant
  // range dropped real contributors on the top/bottom beams. The signed
  // difference to the wedge centre is wrapped into [-pi, pi) so a wedge
  // touching the +-180 deg seam still keeps far-side Gaussians.
  if (USE_AZIM && k) {
    const float x_s = dx * s_fwd[0] + dy * s_fwd[1] + dz * s_fwd[2];
    const float y_s = dx * s_left[0] + dy * s_left[1] + dz * s_left[2];
    const float az = atan2f(y_s, x_s);
    constexpr float kPi = 3.14159265358979323846f;
    // az - az_center is in (-2pi, 2pi); shift by +3pi (always positive),
    // fmod into [0, 2pi), shift back to [-pi, pi).
    const float d = fmodf(az - az_center + 3.0f * kPi, 2.0f * kPi) - kPi;
    const float dist_h = sqrtf(x_s * x_s + y_s * y_s);
    const float ang_margin = margin / fmaxf(dist_h, 1e-6f) + az_pad;
    k = k && (fabsf(d) <= az_halfwidth + ang_margin);
  }

  keep[i] = k;
}

}  // anonymous namespace

// Public entry point: bool mask (torch::kBool) of length N.
torch::Tensor lidar_cull_mask(
    const torch::Tensor& means,          // (N, 3), float32, CUDA
    const torch::Tensor& scales,         // (N, 3), float32, CUDA
    const torch::Tensor& sensor_pos,     // (3,), float32, CUDA — same device
    const torch::Tensor& up_world,       // (3,), float32, CUDA — ignored if !use_elev
    const torch::Tensor& fwd_world,      // (3,), sensor +x in world — ignored if !use_azim
    const torch::Tensor& left_world,     // (3,), sensor +y in world — ignored if !use_azim
    double min_range,
    double max_range,                    // pass negative to disable upper bound
    double cull_scale_sigmas,
    bool use_elev,
    double sin_min,
    double cos_min,
    double sin_max,
    double cos_max,
    bool use_azim,
    double az_center,
    double az_halfwidth,
    double az_pad) {
  TORCH_CHECK(means.is_cuda(), "means must be CUDA");
  TORCH_CHECK(scales.is_cuda(), "scales must be CUDA");
  TORCH_CHECK(means.dtype() == torch::kFloat32, "means must be float32");
  TORCH_CHECK(scales.dtype() == torch::kFloat32, "scales must be float32");
  TORCH_CHECK(means.is_contiguous(), "means must be contiguous");
  TORCH_CHECK(scales.is_contiguous(), "scales must be contiguous");
  TORCH_CHECK(means.dim() == 2 && means.size(1) == 3, "means must be (N, 3)");
  TORCH_CHECK(scales.dim() == 2 && scales.size(1) == 3, "scales must be (N, 3)");
  TORCH_CHECK(means.size(0) == scales.size(0), "means and scales must have same N");

  const int64_t N64 = means.size(0);
  TORCH_CHECK(N64 <= std::numeric_limits<int32_t>::max(),
              "cull kernel supports up to INT32_MAX gaussians");
  const int32_t N = static_cast<int32_t>(N64);

  auto keep = torch::empty({N64}, means.options().dtype(torch::kBool));
  if (N == 0) {
    return keep;
  }

  // sensor_pos / up_world / axes stay on device: the kernel loads them once
  // per block via shared memory. We coerce dtype/device to match `means` so
  // the caller can pass a slice of the 4×4 pose matrix directly.
  auto sensor_dev = sensor_pos.detach().to(means.device(), torch::kFloat32).contiguous();
  auto up_dev = up_world.detach().to(means.device(), torch::kFloat32).contiguous();
  auto fwd_dev = fwd_world.detach().to(means.device(), torch::kFloat32).contiguous();
  auto left_dev = left_world.detach().to(means.device(), torch::kFloat32).contiguous();
  TORCH_CHECK(sensor_dev.numel() == 3, "sensor_pos must have exactly 3 elements");
  TORCH_CHECK(up_dev.numel() == 3, "up_world must have exactly 3 elements");
  TORCH_CHECK(fwd_dev.numel() == 3, "fwd_world must have exactly 3 elements");
  TORCH_CHECK(left_dev.numel() == 3, "left_world must have exactly 3 elements");

  const int block = 256;
  const int grid = static_cast<int>((N + block - 1) / block);

  auto stream = at::cuda::getCurrentCUDAStream();

  auto launch = [&](auto use_elev_c, auto use_azim_c) {
    lidar_cull_kernel<decltype(use_elev_c)::value, decltype(use_azim_c)::value>
        <<<grid, block, 0, stream>>>(
            means.data_ptr<float>(),
            scales.data_ptr<float>(),
            sensor_dev.data_ptr<float>(),
            up_dev.data_ptr<float>(),
            fwd_dev.data_ptr<float>(),
            left_dev.data_ptr<float>(),
            static_cast<float>(min_range),
            static_cast<float>(max_range),
            static_cast<float>(cull_scale_sigmas),
            static_cast<float>(sin_min),
            static_cast<float>(cos_min),
            static_cast<float>(sin_max),
            static_cast<float>(cos_max),
            static_cast<float>(az_center),
            static_cast<float>(az_halfwidth),
            static_cast<float>(az_pad),
            N,
            keep.data_ptr<bool>());
  };
  if (use_elev && use_azim) {
    launch(std::true_type{}, std::true_type{});
  } else if (use_elev) {
    launch(std::true_type{}, std::false_type{});
  } else if (use_azim) {
    launch(std::false_type{}, std::true_type{});
  } else {
    launch(std::false_type{}, std::false_type{});
  }
  AT_CUDA_CHECK(cudaGetLastError());
  return keep;
}

// Bumped whenever a binding's signature OR semantics change: a pre-built .so
// exporting an older version (or none) is stale and must not be used — see
// _lidar_cull_ext. v3: the azimuth wedge margin divides by the horizontal
// distance (v2 divided by slant range and under-kept high-elevation Gaussians).
int64_t cull_ext_abi_version() { return 3; }


// ── Scalar view-dependent raydrop SH evaluation ─────────────────────────
//
// The LiDAR raydrop logit is ONE channel, but gsplat's spherical_harmonics
// takes colour-shaped (N, K, 3) coefficients. Packing into that layout cost a
// throwaway (N, K, 3) buffer -- 306 MiB at N=3.2M, K=9 -- plus its zero-fill
// and two scatter writes, then evaluated 3 channels to use 1. This kernel does
// the same math in a single pass: reads means (12B), the DC logit (4B) and the
// higher bands (4*C_HIGH B) per Gaussian and writes one float.
//
// Convention matches gsplat exactly: dir = normalize(mean - view_pos), the DC
// term is the scalar logit (i.e. coefficient logit/SH_C0 scaled by SH_C0), and
// the higher bands follow gsplat's coefficient order.

namespace {

__device__ __forceinline__ float sh_eval_scalar(
    const float* __restrict__ high,  // (C_HIGH,) higher bands for this Gaussian
    const int degree,
    const float dc,                  // band-0 contribution (the scalar logit)
    const float x, const float y, const float z) {
  // gsplat's SH_C* constants.
  constexpr float C1 = 0.4886025119029199f;
  constexpr float C20 = 1.0925484305920792f;
  constexpr float C21 = -1.0925484305920792f;
  constexpr float C22 = 0.31539156525252005f;
  constexpr float C23 = -1.0925484305920792f;
  constexpr float C24 = 0.5462742152960396f;
  constexpr float C30 = -0.5900435899266435f;
  constexpr float C31 = 2.890611442640554f;
  constexpr float C32 = -0.4570457994644658f;
  constexpr float C33 = 0.3731763325901154f;
  constexpr float C34 = -0.4570457994644658f;
  constexpr float C35 = 1.445305721320277f;
  constexpr float C36 = -0.5900435899266435f;

  float out = dc;
  if (degree < 1) return out;
  out += C1 * (-y * high[0] + z * high[1] - x * high[2]);
  if (degree < 2) return out;
  const float xx = x * x, yy = y * y, zz = z * z;
  const float xy = x * y, yz = y * z, xz = x * z;
  out += C20 * xy * high[3]
       + C21 * yz * high[4]
       + C22 * (2.f * zz - xx - yy) * high[5]
       + C23 * xz * high[6]
       + C24 * (xx - yy) * high[7];
  if (degree < 3) return out;
  out += C30 * y * (3.f * xx - yy) * high[8]
       + C31 * xy * z * high[9]
       + C32 * y * (4.f * zz - xx - yy) * high[10]
       + C33 * z * (2.f * zz - 3.f * xx - 3.f * yy) * high[11]
       + C34 * x * (4.f * zz - xx - yy) * high[12]
       + C35 * z * (xx - yy) * high[13]
       + C36 * x * (xx - 3.f * yy) * high[14];
  return out;
}

template <int C_HIGH>
__global__ void raydrop_sh_kernel(
    const float* __restrict__ means,        // (N, 3)
    const float* __restrict__ view_pos,     // (3,)
    const float* __restrict__ dc_logit,     // (N,)
    const float* __restrict__ high,         // (N, C_HIGH)
    const int degree,
    const int64_t N,
    float* __restrict__ out) {              // (N,)
  __shared__ float s_view[3];
  if (threadIdx.x < 3) s_view[threadIdx.x] = view_pos[threadIdx.x];
  __syncthreads();

  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= N) return;

  float dx = means[i * 3 + 0] - s_view[0];
  float dy = means[i * 3 + 1] - s_view[1];
  float dz = means[i * 3 + 2] - s_view[2];
  // gsplat normalizes the direction before evaluating the basis.
  const float inv = rsqrtf(fmaxf(dx * dx + dy * dy + dz * dz, 1e-20f));
  dx *= inv; dy *= inv; dz *= inv;

  out[i] = sh_eval_scalar(high + i * C_HIGH, degree, dc_logit[i], dx, dy, dz);
}

}  // namespace

torch::Tensor raydrop_sh_eval(
    const torch::Tensor& means,          // (N, 3) float32 CUDA
    const torch::Tensor& view_pos,       // (3,) float32 CUDA
    const torch::Tensor& dc_logit,       // (N,) float32 CUDA
    const torch::Tensor& high) {         // (N, C_HIGH) float32 CUDA
  TORCH_CHECK(means.is_cuda() && means.scalar_type() == at::kFloat);
  TORCH_CHECK(means.dim() == 2 && means.size(1) == 3, "means must be (N, 3)");
  TORCH_CHECK(high.dim() == 2 && high.size(0) == means.size(0),
              "high must be (N, C_HIGH) matching means");
  TORCH_CHECK(dc_logit.dim() == 1 && dc_logit.size(0) == means.size(0));
  TORCH_CHECK(view_pos.numel() == 3);

  auto means_c = means.contiguous();
  auto high_c = high.contiguous();
  auto dc_c = dc_logit.contiguous();
  auto vp_c = view_pos.to(means.options()).contiguous().view({3});

  const int64_t N = means_c.size(0);
  const int64_t c_high = high_c.size(1);
  // c_high == (degree + 1)^2 - 1
  int degree = -1;
  for (int d = 0; d <= 3; ++d) {
    if ((d + 1) * (d + 1) - 1 == c_high) { degree = d; break; }
  }
  TORCH_CHECK(degree >= 0, "raydrop_sh width ", c_high,
              " is not (deg+1)^2-1 for deg in [0, 3]");

  auto out = at::empty({N}, means_c.options());
  if (N == 0) return out;

  const int block = 256;
  const int grid = static_cast<int>((N + block - 1) / block);
  auto stream = at::cuda::getCurrentCUDAStream();

  const float* m = means_c.data_ptr<float>();
  const float* v = vp_c.data_ptr<float>();
  const float* d = dc_c.data_ptr<float>();
  const float* h = high_c.data_ptr<float>();
  float* o = out.data_ptr<float>();

  switch (c_high) {
    case 3:
      raydrop_sh_kernel<3><<<grid, block, 0, stream>>>(m, v, d, h, degree, N, o);
      break;
    case 8:
      raydrop_sh_kernel<8><<<grid, block, 0, stream>>>(m, v, d, h, degree, N, o);
      break;
    case 15:
      raydrop_sh_kernel<15><<<grid, block, 0, stream>>>(m, v, d, h, degree, N, o);
      break;
    default:
      TORCH_CHECK(false, "unsupported raydrop_sh width ", c_high);
  }
  AT_CUDA_CHECK(cudaGetLastError());
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "lidar_cull_mask",
      &lidar_cull_mask,
      "Fused frustum + elevation-FOV + azimuth-wedge cull for LiDAR (CUDA)",
      pybind11::arg("means"),
      pybind11::arg("scales"),
      pybind11::arg("sensor_pos"),
      pybind11::arg("up_world"),
      pybind11::arg("fwd_world"),
      pybind11::arg("left_world"),
      pybind11::arg("min_range"),
      pybind11::arg("max_range"),
      pybind11::arg("cull_scale_sigmas"),
      pybind11::arg("use_elev"),
      pybind11::arg("sin_min"),
      pybind11::arg("cos_min"),
      pybind11::arg("sin_max"),
      pybind11::arg("cos_max"),
      pybind11::arg("use_azim") = false,
      pybind11::arg("az_center") = 0.0,
      pybind11::arg("az_halfwidth") = 0.0,
      pybind11::arg("az_pad") = 0.0);
  m.def(
      "cull_ext_abi_version",
      &cull_ext_abi_version,
      "ABI version of this extension build (staleness guard)");
  m.def(
      "raydrop_sh_eval",
      &raydrop_sh_eval,
      "Scalar view-dependent raydrop SH evaluation (CUDA)",
      pybind11::arg("means"),
      pybind11::arg("view_pos"),
      pybind11::arg("dc_logit"),
      pybind11::arg("high"));
}
