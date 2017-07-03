/**
 * @file Update_device.cu
 * @date 2017-05-30
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "Update.h"
#include "Cuda.h"

template <typename Dtype>
__global__ void DoNesterov(int size, const Dtype* dx, Dtype* v_prev, Dtype* v, Dtype* x,
    const Dtype mu, const Dtype lr)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

    /****
     * Nesterov Alogorithm
     *
     * v_prev = v # back this up
     * v = mu * v - learning_rate * dx # velocity update stays the same
     * x += -mu * v_prev + (1 + mu) * v # position update changes form
     *
     */

    v_prev[idx] = v[idx];
    v[idx] = mu * v[idx] - lr * dx[idx];
    x[idx] += (-1.0) * mu * v_prev[idx] + (1 + mu) * v[idx];
}

template <typename Dtype>
__global__ void DoAdagrad(int size, const Dtype* dx, Dtype* cache, Dtype* x,
    const Dtype lr, const Dtype eps)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

    /****
     * Adagrad Alogorithm
     *
     * cache += dx**2
     * x += -learning_rate * dx / (sqrt(cache) + eps)
     *
     */

    cache[idx] += dx[idx] * dx[idx];
    x[idx] += (-1.0) * lr * dx[idx] / (sqrtf(cache[idx]) + eps);
}

template <typename Dtype>
__global__ void DoRMSprop(int size, const Dtype* dx, Dtype* cache, Dtype* x,
    const Dtype lr, const Dtype eps, const Dtype dr)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

    /****
     * RMSprop
     *
     * cache = decay_rate * cache + (1 - decay_rate) * dx**2
     * x += - learning_rate * dx / (sqrt(cache) + eps)
     *
     */

    cache[idx] = dr * cache[idx] + (1.0 - dr) * dx[idx] * dx[idx];
    x[idx] += (-1.0) * lr * dx[idx] / (sqrtf(cache[idx]) + eps);
}

#define USE_TENSORFLOW_ADAM         0

template <typename Dtype>
__global__ void DoAdam(int size, const Dtype* dx, Dtype* m, Dtype* v, Dtype* x,
    const Dtype lr, const Dtype eps, const Dtype beta1, const Dtype beta2,
    const Dtype decayedBeta1, const Dtype decayedBeta2)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

    /****
     * Adam
     *
     * m = beta1 * m + (1 - beta1) * dx
     * v = beta2 * v + (1 - beta2) * (dx**2)
     * x += -learning_rate * m / (sqrt(v) + eps)
     *
     */
    m[idx] = beta1 * m[idx] + (1.0 - beta1) * dx[idx];
    v[idx] = beta2 * v[idx] + (1.0 - beta2) * dx[idx] * dx[idx];
#if USE_TENSORFLOW_ADAM
    Dtype learningRate = lr * sqrtf(1.0 - decayedBeta2) / (1.0 - decayedBeta1);
    x[idx] += (-1.0) * learningRate * m[idx] / (sqrtf(v[idx]) + eps);
#else
    x[idx] += (-1.0) * lr * m[idx] / (sqrtf(v[idx]) + eps);

#endif
}

template<typename Dtype>
void Update<Dtype>::doNesterov(int size, const Dtype* dx, Dtype* v_prev, Dtype* v, Dtype* x,
    const Dtype mu, const Dtype lr) {
    DoNesterov<<<SOOOA_GET_BLOCKS(size), SOOOA_CUDA_NUM_THREADS>>>(
        size, dx, v_prev, v, x, mu, lr);
}

template<typename Dtype>
void Update<Dtype>::doAdagrad(int size, const Dtype* dx, Dtype* cache, Dtype* x,
    const Dtype lr, const Dtype eps) {
    DoAdagrad<<<SOOOA_GET_BLOCKS(size), SOOOA_CUDA_NUM_THREADS>>>(
        size, dx, cache, x, lr, eps);
}

template<typename Dtype>
void Update<Dtype>::doRMSprop(int size, const Dtype* dx, Dtype* cache, Dtype* x,
    const Dtype lr, const Dtype eps, const Dtype dr) {

    DoRMSprop<<<SOOOA_GET_BLOCKS(size), SOOOA_CUDA_NUM_THREADS>>>(
        size, dx, cache, x, lr, eps, dr); 
}

template<typename Dtype>
void Update<Dtype>::doAdam(int size, const Dtype* dx, Dtype* m, Dtype* v, Dtype* x,
    const Dtype lr, const Dtype eps, const Dtype beta1, const Dtype beta2,
    const Dtype decayedBeta1, const Dtype decayedBeta2) {

    DoAdam<<<SOOOA_GET_BLOCKS(size), SOOOA_CUDA_NUM_THREADS>>>(
        size, dx, m, v, x, lr, eps, beta1, beta2, decayedBeta1, decayedBeta2);
}


template void Update<float>::doNesterov(int size, const float* dx, float* v_prev,
        float* v, float* x, const float mu, const float lr);
template void Update<float>::doAdagrad(int size, const float* dx, float* cache,
        float* x, const float lr, const float eps);
template void Update<float>::doRMSprop(int size, const float* dx, float* cache,
        float* x, const float lr, const float eps, const float dr);
template void Update<float>::doAdam(int size, const float* dx, float* m,
        float* v, float* x, const float lr, const float eps, const float beta1,
        const float beta2, const float decayedBeta1, const float decayedBeta2);
