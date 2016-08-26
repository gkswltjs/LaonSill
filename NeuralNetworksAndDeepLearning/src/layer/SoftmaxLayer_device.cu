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
__global__ void Dropout_(const int n, const DATATYPE* in, const DATATYPE* mask,
		const unsigned int threashold, const float scale, DATATYPE *out)
{

	CUDA_KERNEL_LOOP(index, n) {
		//out[index] = in[index] * (mask[index] > threshold) * scale;
		out[index] = in[index] * (mask[index]) * scale;
	}
}



void SoftmaxLayer::backpropagation(const uint32_t* target) {
	Util::printMessage("SoftmaxLayer::target()---"+string(name));

	_target.set_mem(target, SyncMemCopyType::HostToDevice);
	//_target.print("target:");

	const DATATYPE* d_z = _preActivation->device_data();
	const DATATYPE* d_output = _output->device_data();
	const uint32_t* d_target = _target.device_mem();
	//DATATYPE* d_delta = _preActivation->mutable_device_grad();
	_output->reset_device_grad();
	DATATYPE* d_delta = _output->mutable_device_grad();
	cost_fn->backward(d_z, d_output, d_target, d_delta, out_dim.rows, out_dim.batches);

	//Util::printDeviceData(d_delta, out_dim.rows, out_dim.batches, 1, 1, "d_delta:");
	_output->print_data("d_output:");
	//_target.print("d_target:");
	_output->print_grad("d_delta:");

	_backpropagation();
	propBackpropagation();


	//_output->reset_device_grad();
	//OutputLayer::backpropagation(id, getInput(), 0);


	// Accounting for batch size in SGD
	// checkCudaErrors(cublasSscal(cublasHandle, ref_fc2.outputs * m_batchSize, &scalVal, dloss_data, 1));

	/*
	if(Util::train && p_dropout < 1.0f) {
		//Util::setPrint(true);
		Util::printDeviceData(d_delta, out_dim.rows, out_dim.batches, 1, 1, "delta_input:");
		Dropout_<<<RoundUp(out_dim.batchsize(), BW), BW>>>(out_dim.batchsize(), d_delta, d_mask, 0, scale, d_delta);


		Util::printData(mask, out_dim.rows, out_dim.batches, 1, 1, this->name+string("/mask:"));
		//DATATYPE *next_delta_input = next_layer->getDeltaInput();
		Util::printDeviceData(d_delta, out_dim.rows, out_dim.batches, 1, 1, "delta_input:");
		//Util::setPrint(false);
	}
	*/

	//Util::printDeviceData(d_input, in_dim.rows, in_dim.batches, 1, 1, "d_input:");

	/*
	_input->print_data("d_input:");
	const DATATYPE* d_input = _input->device_data();
	DATATYPE* d_delta_weight = _params[Weight]->mutable_device_grad();
	checkCudaErrors(cublasSgemm(Cuda::cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, out_dim.rows, in_dim.rows, out_dim.batches,
			&Cuda::alpha, d_delta, out_dim.rows, d_input, in_dim.rows, &Cuda::beta, d_delta_weight, out_dim.rows));
	//Util::printDeviceData(d_delta_weight, out_dim.rows, in_dim.rows, 1, 1, "d_delta_weight:");
	_params[Weight]->print_grad("d_delta_weight:");

	DATATYPE* d_delta_bias = _params[Bias]->mutable_device_grad();
	checkCudaErrors(cublasSgemv(Cuda::cublasHandle, CUBLAS_OP_N, out_dim.rows, out_dim.batches,
			&Cuda::alpha, d_delta, out_dim.rows, d_onevec, 1, &Cuda::beta, d_delta_bias, 1));
	//Util::printDeviceData(d_delta_bias, out_dim.rows, 1, 1, 1, "d_delta_bias:");
	_params[Bias]->print_grad("d_delta_bias:");

	//Util::printDeviceData(d_weight, out_dim.rows, in_dim.rows, 1, 1, "d_weight:");
	//Util::printDeviceData(d_delta, out_dim.rows, out_dim.batches, 1, 1, "d_delta:");
	_params[Weight]->print_data("d_weight:");
	_preActivation->print_grad("d_delta");

	const DATATYPE* d_weight = _params[Weight]->device_data();
	DATATYPE* d_delta_input = _input->mutable_device_grad();
	checkCudaErrors(cublasSgemm(Cuda::cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, in_dim.rows, out_dim.batches, out_dim.rows,
			&Cuda::alpha, d_weight, out_dim.rows, d_delta, out_dim.rows, &Cuda::beta, d_delta_input, in_dim.rows));

	//Util::printDeviceData(d_delta_input, in_dim.rows, in_dim.batches, 1, 1, "d_delta_input:");
	_input->print_grad("d_delta_input:");

	propBackpropagation();
	*/
}


double SoftmaxLayer::cost(const uint32_t* target) {
	// 편의상 HOST에서 계산, DEVICE 코드로 변환해야 함
	_target.set_mem(target, SyncMemCopyType::HostToHost);
	return cost_fn->forward(_output->host_data(), _target.host_mem(), out_dim.rows, out_dim.batches);
}


#endif








