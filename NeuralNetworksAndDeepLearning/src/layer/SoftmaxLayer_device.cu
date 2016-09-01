/*
 * SoftmaxLayer.cpp
 *
 *  Created on: 2016. 8. 1.
 *      Author: jhkim
 */


#ifdef GPU_MODE

#include "SoftmaxLayer.h"

///////////////////////////////////////////////////////////////////////////////////////////
// GPU Kernels

/**
 * Fills a floating-point array with ones.
 *
 * @param vec The array to fill.
 * @param size The number of elements in the array.
 */
template <typename Dtype>
__global__ void Dropout_(const int n, const Dtype* in, const Dtype* mask,
		const uint32_t threashold, const float scale, Dtype *out)
{

	CUDA_KERNEL_LOOP(index, n) {
		//out[index] = in[index] * (mask[index] > threshold) * scale;
		out[index] = in[index] * (mask[index]) * scale;
	}
}


template <typename Dtype>
void SoftmaxLayer<Dtype>::backpropagation(const uint32_t* target) {
	Util::printMessage("SoftmaxLayer::target()---"+string(this->name));
	this->_target.set_mem(target, SyncMemCopyType::HostToDevice);

	const Dtype* d_preActivationData = this->_preActivation->device_data();
	const Dtype* d_outputData = this->_output->device_data();
	const uint32_t* d_targetData = this->_target.device_mem();

	// delta_output 구하는 단계를 넣을 경우, delta_output을 0으로 reset할 필요가 있음
	// 0으로 reset한 후, target에 해당하는 element만 수정, (테스트 단계 임시로 여기서 reset)
	//this->_output->reset_device_grad();
	//Dtype* d_outputGrad = this->_output->mutable_device_grad();
	Dtype* d_preActivationGrad = this->_preActivation->mutable_device_grad();

	this->cost_fn->backward(d_preActivationData, d_outputData, d_targetData, d_preActivationGrad,
			this->out_dim.rows, this->out_dim.batches);

	/*
	Data<Dtype>::printConfig = 1;
	this->_output->print_data("d_outputData:");
	this->_target.print("d_targetData:");
	//this->_output->print_grad("d_delta:");
	this->_preActivation->print_grad("d_preActivationGrad:");
	Data<Dtype>::printConfig = 0;
	*/


	/*
	if(this->_output->is_inf_grad()) {
		cout << this->name << " output gradient inf ... skip this batch ... " << endl;
		return;

		//Data<Dtype>::printConfig = 1;
		//this->_output->print_data("out:");
		//this->_output->print_grad("deltaOut:");
		//Data<Dtype>::printConfig = 0;
		//exit(1);
	}
	*/




	//OutputLayer<Dtype>::_activationBackward();
	OutputLayer<Dtype>::_backpropagation();
	OutputLayer<Dtype>::propBackpropagation();


	//_output->reset_device_grad();
	//OutputLayer<Dtype>::backpropagation(id, getInput(), 0);


	// Accounting for batch size in SGD
	// checkCudaErrors(cublasSscal(cublasHandle, ref_fc2.outputs * m_batchSize, &scalVal, dloss_data, 1));

	/*
	if(Util::train && p_dropout < 1.0f) {
		//Util::setPrint(true);
		Util::printDeviceData(d_delta, out_dim.rows, out_dim.batches, 1, 1, "delta_input:");
		Dropout_<<<RoundUp(out_dim.batchsize(), BW), BW>>>(out_dim.batchsize(), d_delta, d_mask, 0, scale, d_delta);


		Util::printData(mask, out_dim.rows, out_dim.batches, 1, 1, this->name+string("/mask:"));
		//Dtype *next_delta_input = next_layer->getDeltaInput();
		Util::printDeviceData(d_delta, out_dim.rows, out_dim.batches, 1, 1, "delta_input:");
		//Util::setPrint(false);
	}
	*/

	//Util::printDeviceData(d_input, in_dim.rows, in_dim.batches, 1, 1, "d_input:");

	/*
	_input->print_data("d_input:");
	const Dtype* d_input = _input->device_data();
	Dtype* d_delta_weight = _params[Weight]->mutable_device_grad();
	checkCudaErrors(cublasSgemm(Cuda::cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, out_dim.rows, in_dim.rows, out_dim.batches,
			&Cuda::alpha, d_delta, out_dim.rows, d_input, in_dim.rows, &Cuda::beta, d_delta_weight, out_dim.rows));
	//Util::printDeviceData(d_delta_weight, out_dim.rows, in_dim.rows, 1, 1, "d_delta_weight:");
	_params[Weight]->print_grad("d_delta_weight:");

	Dtype* d_delta_bias = _params[Bias]->mutable_device_grad();
	checkCudaErrors(cublasSgemv(Cuda::cublasHandle, CUBLAS_OP_N, out_dim.rows, out_dim.batches,
			&Cuda::alpha, d_delta, out_dim.rows, d_onevec, 1, &Cuda::beta, d_delta_bias, 1));
	//Util::printDeviceData(d_delta_bias, out_dim.rows, 1, 1, 1, "d_delta_bias:");
	_params[Bias]->print_grad("d_delta_bias:");

	//Util::printDeviceData(d_weight, out_dim.rows, in_dim.rows, 1, 1, "d_weight:");
	//Util::printDeviceData(d_delta, out_dim.rows, out_dim.batches, 1, 1, "d_delta:");
	_params[Weight]->print_data("d_weight:");
	_preActivation->print_grad("d_delta");

	const Dtype* d_weight = _params[Weight]->device_data();
	Dtype* d_delta_input = _input->mutable_device_grad();
	checkCudaErrors(cublasSgemm(Cuda::cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, in_dim.rows, out_dim.batches, out_dim.rows,
			&Cuda::alpha, d_weight, out_dim.rows, d_delta, out_dim.rows, &Cuda::beta, d_delta_input, in_dim.rows));

	//Util::printDeviceData(d_delta_input, in_dim.rows, in_dim.batches, 1, 1, "d_delta_input:");
	_input->print_grad("d_delta_input:");

	propBackpropagation();
	*/
}

template <typename Dtype>
double SoftmaxLayer<Dtype>::cost(const uint32_t* target) {
	// 편의상 HOST에서 계산, DEVICE 코드로 변환해야 함
	this->_target.set_mem(target, SyncMemCopyType::HostToHost);
	return this->cost_fn->forward(this->_output->host_data(), this->_target.host_mem(), this->out_dim.rows, this->out_dim.batches);
}


template void SoftmaxLayer<float>::backpropagation(const uint32_t* target);
template double SoftmaxLayer<float>::cost(const uint32_t* target);



#endif








