#include "MathFunctions.h"
#include "Cuda.h"

template <typename Dtype>
__global__ void sub_if_ge_than_zero(const uint32_t n, const Dtype* f,
    const Dtype* p, Dtype* df) {
  CUDA_KERNEL_LOOP(index, n) {
	  if (f[index] > 0)
		  df[index] = f[index] - p[index];
	  else
		  df[index] = 0;
  }
}


template <typename Dtype>
void diff_content_loss(const uint32_t n, const Dtype* f,
    const Dtype* p, Dtype* df) {
	sub_if_ge_than_zero<Dtype><<<SOOOA_GET_BLOCKS(n), SOOOA_CUDA_NUM_THREADS>>>(
		n, f, p, df);
}

template void diff_content_loss<float>(const uint32_t n, const float* f,
		const float* p, float* df);





template <typename Dtype>
__global__ void ignore_if_le_than_zero(const uint32_t n, const Dtype* f, Dtype* a) {

	CUDA_KERNEL_LOOP(index, n) {
		if (f[index] < 0)
			a[index] = 0;
	}
}

template <typename Dtype>
void diff_style_loss(const uint32_t n, const Dtype* f, Dtype* a) {
	ignore_if_le_than_zero<Dtype><<<SOOOA_GET_BLOCKS(n), SOOOA_CUDA_NUM_THREADS>>>(
			n, f, a);
}

template void diff_style_loss<float>(const uint32_t n, const float* f, float* a);
