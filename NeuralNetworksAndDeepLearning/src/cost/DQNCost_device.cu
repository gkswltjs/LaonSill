/**
 * @file DQNCost_device.cu
 * @date 2016-12-27
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "DQNCost.h"

using namespace std;


#ifdef GPU_MODE
template <typename Dtype>
__global__ void DQNLossBackprop(
		const Dtype* z,
		const Dtype* activation,
		const Dtype* target,
		Dtype *delta,
		uint32_t totalSize) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= totalSize) return;

	delta[idx] = (activation[idx] - target[idx]);
}
#endif

#ifndef GPU_MODE
template <typename Dtype>
double DQNCost<Dtype>::fn(const rvec *pA, const rvec *pY) {
	return 0.0;
}

template <typename Dtype>
void DQNCost<Dtype>::d_cost(const rcube &z, const rcube &activation, const rvec &target,
    rcube &delta) {
	Util::printCube(activation, "activation:");
	Util::printVec(target, "target:");

	delta.slice(0) = activation.slice(0) - target;
	Util::printCube(delta, "delta:");
}

#else

template <typename Dtype>
double DQNCost<Dtype>::forward(const Dtype* output, const Dtype* target,
    const uint32_t numLabels, const uint32_t batchsize) {
	double cost = 0.0;


	for (uint32_t batchIndex = 0; batchIndex < batchsize; batchIndex++) {
        for (uint32_t labelIndex = 0; labelIndex < numLabels; labelIndex++) {
            uint32_t index = batchIndex * numLabels + labelIndex;
            cost += pow((output[index] - target[index]), 2.0);
        }
	}
	return cost;
}

template <typename Dtype>
void DQNCost<Dtype>::backward(const Dtype* z, const Dtype* activation,
    const Dtype* target, Dtype* delta, uint32_t numLabels, uint32_t batchsize) {

    int totalSize = numLabels * batchsize;
	DQNLossBackprop<<<RoundUp(totalSize, BW), BW>>>(z, activation, target, delta, totalSize);
	CUDA_POST_KERNEL_CHECK;
}


template class DQNCost<float>;

#endif
