#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void ink_density_forward_kernel(
    const float* __restrict__ image,
    const int64_t* __restrict__ widths,
    float* __restrict__ sum,
    int* __restrict__ count,
    const int n,
    const int c,
    const int h,
    const int w) {
  const int total = n * c * h * w;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }

  const int x = index % w;
  const int sample = index / (c * h * w);
  const int width = max(1, min(static_cast<int>(widths[sample]), w));
  if (x < width) {
    const float ink = fminf(fmaxf((1.0f - image[index]) * 0.5f, 0.0f), 1.0f);
    atomicAdd(sum, ink);
    atomicAdd(count, 1);
  }
}

__global__ void ink_density_backward_kernel(
    float* __restrict__ grad_image,
    const int64_t* __restrict__ widths,
    const float grad_output,
    const float denom,
    const int n,
    const int c,
    const int h,
    const int w) {
  const int total = n * c * h * w;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }

  const int x = index % w;
  const int sample = index / (c * h * w);
  const int width = max(1, min(static_cast<int>(widths[sample]), w));
  grad_image[index] = x < width ? -0.5f * grad_output / denom : 0.0f;
}

torch::Tensor ink_density_forward(torch::Tensor image, torch::Tensor widths) {
  TORCH_CHECK(image.is_cuda(), "image must be a CUDA tensor");
  TORCH_CHECK(widths.is_cuda(), "widths must be a CUDA tensor");
  TORCH_CHECK(image.dtype() == torch::kFloat32, "image must be float32");
  TORCH_CHECK(image.dim() == 4, "image must be NCHW");

  auto sum = torch::zeros({1}, image.options());
  auto count = torch::zeros({1}, torch::TensorOptions().device(image.device()).dtype(torch::kInt32));
  const int n = image.size(0);
  const int c = image.size(1);
  const int h = image.size(2);
  const int w = image.size(3);
  const int total = n * c * h * w;
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;
  ink_density_forward_kernel<<<blocks, threads>>>(
      image.data_ptr<float>(),
      widths.data_ptr<int64_t>(),
      sum.data_ptr<float>(),
      count.data_ptr<int>(),
      n, c, h, w);
  auto denom = count.to(image.dtype()).clamp_min(1);
  return sum / denom;
}

torch::Tensor ink_density_backward(
    torch::Tensor grad_output,
    torch::Tensor widths,
    double denom,
    std::vector<int64_t> shape,
    torch::Device device) {
  auto grad_image = torch::empty(shape, torch::TensorOptions().device(device).dtype(torch::kFloat32));
  const int n = shape[0];
  const int c = shape[1];
  const int h = shape[2];
  const int w = shape[3];
  const int total = n * c * h * w;
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;
  ink_density_backward_kernel<<<blocks, threads>>>(
      grad_image.data_ptr<float>(),
      widths.data_ptr<int64_t>(),
      grad_output.item<float>(),
      static_cast<float>(denom),
      n, c, h, w);
  return grad_image;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &ink_density_forward, "Ink density forward (CUDA)");
  m.def("backward", &ink_density_backward, "Ink density backward (CUDA)");
}
