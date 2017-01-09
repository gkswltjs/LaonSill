/**
 * @file DQNOutputLayer_device.cu
 * @date 2016-12-26
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "DQNOutputLayer.h"

using namespace std;

///////////////////////////////////////////////////////////////////////////////////////////
// GPU Kernels

/**
 * Fills a floating-point array with ones.
 *
 * @param vec The array to fill.
 * @param size The number of elements in the array.
 */
/*
template <typename Dtype>
__global__ void Dropout_(const int n, const Dtype* in, const Dtype* mask,
		const uint32_t threashold, const float scale, Dtype *out) {
	CUDA_KERNEL_LOOP(index, n) {
		//out[index] = in[index] * (mask[index] > threshold) * scale;
		out[index] = in[index] * (mask[index]) * scale;
	}
}
*/

template <typename Dtype>
void DQNOutputLayer<Dtype>::backpropagation() {
	const Dtype* d_preActivationData = this->_preActivation->device_data();
	const Dtype* d_outputData = this->_outputData[0]->device_data();
	const Dtype* d_target = this->_inputData[1]->device_data();

	// delta_output 구하는 단계를 넣을 경우, delta_output을 0으로 reset할 필요가 있음
	// 0으로 reset한 후, target에 해당하는 element만 수정, (테스트 단계 임시로 여기서 reset)
	this->_outputData[0]->reset_device_grad();
	Dtype* d_outputGrad = this->_outputData[0]->mutable_device_grad();

    int batchSize = this->_outputData[0]->getShape(0);
    int rowSize = this->_outputData[0]->getShape(2);

	this->cost_fn->backward(d_preActivationData, d_outputData, d_target, 
        d_outputGrad, rowSize, batchSize);

	OutputLayer<Dtype>::backpropagation();
}

template <typename Dtype>
//double DQNOutputLayer<Dtype>::cost(const uint32_t* target) {
double DQNOutputLayer<Dtype>::cost() {
	// 편의상 HOST에서 계산, DEVICE 코드로 변환해야 함
	//this->_target.set_mem(target, SyncMemCopyType::HostToHost);

	if (this->_inputs.size() <= 1) {
		cout << "DQNOutputLayer does not have label data ... " << endl;
		exit(1);
	}

	const Dtype* h_outputData = this->_outputData[0]->host_data();
	const Dtype* h_target = this->_inputData[1]->host_data();

    int batchSize = this->_outputData[0]->getShape(0);
    int rowSize = this->_outputData[0]->getShape(2);
	return this->cost_fn->forward(h_outputData, h_target, rowSize, batchSize);
}


//template void DQNOutputLayer<float>::backpropagation(const uint32_t* target);
template void DQNOutputLayer<float>::backpropagation();
//template double DQNOutputLayer<float>::cost(const uint32_t* target);
template double DQNOutputLayer<float>::cost();
