/*
 * FullyConnectedLayer.cpp
 *
 *  Created on: 2016. 5. 10.
 *      Author: jhkim
 */

#include "FullyConnectedLayer.h"
#include "../Util.h"
#include "../exception/Exception.h"
#include "../network/NetworkConfig.h"


#ifdef GPU_MODE
///////////////////////////////////////////////////////////////////////////////////////////
// GPU Kernels

/**
 * Fills a floating-point array with ones.
 *
 * @param vec The array to fill.
 * @param size The number of elements in the array.
 */
template <typename Dtype>
__global__ void FillValues(Dtype *vec, int size, Dtype value)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;
	vec[idx] = value;
}


///////////////////////////////////////////////////////////////////////////////////////////
// GPU Kernels

/**
 * Fills a floating-point array with ones.
 *
 * @param vec The array to fill.
 * @param size The number of elements in the array.
 */
template <typename Dtype>
__global__ void Dropout(const int n, const Dtype* in, const Dtype* mask,
		const unsigned int threashold, const float scale, Dtype *out)
{

	CUDA_KERNEL_LOOP(index, n) {
		//out[index] = in[index] * (mask[index] > threshold) * scale;
		out[index] = in[index] * (mask[index]) * scale;
	}
}




template <typename Dtype>
FullyConnectedLayer<Dtype>::~FullyConnectedLayer() {
	delete _params[ParamType::Weight];
	delete _params[ParamType::Bias];
	_params.clear();

	delete _paramsHistory[ParamType::Weight];
	delete _paramsHistory[ParamType::Bias];
	_paramsHistory.clear();

	delete _preActivation;
	checkCudaErrors(cudaFree(d_onevec));

	ActivationFactory<Dtype>::destory(activation_fn);
}


template <typename Dtype>
void FullyConnectedLayer<Dtype>::_shape(bool recursive) {
	this->in_dim.rows = this->in_dim.rows*this->in_dim.cols*this->in_dim.channels;
	this->in_dim.cols = 1;
	this->in_dim.channels = 1;
	this->out_dim.batches = this->in_dim.batches;

	if(recursive) {
		HiddenLayer<Dtype>::_shape();
	}

	uint32_t u_in = this->in_dim.unitsize();
	uint32_t u_out = this->out_dim.unitsize();
	uint32_t b_in = this->in_dim.batchsize();
	uint32_t b_out = this->out_dim.batchsize();

	//weight = new Dtype[u_out*u_in];
	//bias = new Dtype[u_out];
	_params[ParamType::Weight]->reshape({1, 1, u_out, u_in});
	_params[ParamType::Bias]->reshape({1, 1, u_out, 1});
	_paramsHistory[ParamType::Weight]->reshape({1, 1, u_out, u_in});
	_paramsHistory[ParamType::Bias]->reshape({1, 1, u_out, 1});
	_preActivation->reshape({this->out_dim.batches, 1, u_out, 1});


	//cout << this->name << ", fanin: " << u_out*u_in << endl;
	weight_filler.fill(_params[ParamType::Weight]->mutable_host_data(), u_out*u_in, u_in, u_out);
	bias_filler.fill(_params[ParamType::Bias]->mutable_host_data(), u_out, u_in, u_out);

	_params[ParamType::Weight]->print_data("weight:");
	_params[ParamType::Bias]->print_data("bias:");

	checkCudaErrors(Util::ucudaMalloc(&this->d_onevec, sizeof(Dtype)*this->in_dim.batches));
	FillValues<<<RoundUp(this->in_dim.batches, BW), BW>>>(this->d_onevec, this->in_dim.batches, 1.0f);
	//cuda_FillValues(this->d_onevec, in_dim.batches, 1.0f);
	//checkCudaErrors(cudaMemset(d_onevec, 1, in_dim.batches));


	//checkCudaErrors(cudaMemcpyAsync(this->d_weight, weight, sizeof(Dtype)*u_out*u_in, cudaMemcpyHostToDevice));
	//checkCudaErrors(cudaMemcpyAsync(this->d_bias, bias, sizeof(Dtype)*u_out, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaDeviceSynchronize());

	mask = new Dtype[b_out];
	checkCudaErrors(Util::ucudaMalloc(&this->d_mask, sizeof(Dtype)*b_out));
}

template <typename Dtype>
void FullyConnectedLayer<Dtype>::_clearShape() {
	delete _params[0];
	delete _params[1];
	//_params.clear();

	delete _paramsHistory[0];
	delete _paramsHistory[1];
	//_paramsHistory.clear();

	delete _preActivation;


	if(mask) delete [] mask;
	checkCudaErrors(cudaFree(d_mask));
	HiddenLayer<Dtype>::_clearShape();
}


template <typename Dtype>
void FullyConnectedLayer<Dtype>::update() {
	/*
	//weight = (1-eta*lambda/n)*weight - (eta/miniBatchSize)*nabla_w;
	//bias -= eta/miniBatchSize*nabla_b;
	//weight = (1-weight_update_param.lr_mult*weight_update_param.decay_mult/n)*weight - (weight_update_param.lr_mult/miniBatchSize)*nabla_w;
	//bias -= bias_update_param.lr_mult/miniBatchSize*nabla_b;

	float delta_scale = -weight_update_param.lr_mult/miniBatchSize;
	float param_scale = 1-weight_update_param.lr_mult*weight_update_param.decay_mult/n;
	float b_delta_scale = -bias_update_param.lr_mult/miniBatchSize;

	Util::printDeviceData(d_delta_weight, out_dim.rows, in_dim.rows, 1, 1, "delta_weight:");
	Util::printDeviceData(d_weight, out_dim.rows, in_dim.rows, 1, 1, "weight:");
	checkCudaErrors(cublasSscal(Cuda::cublasHandle, static_cast<int>(in_dim.rows*out_dim.rows), &param_scale, d_weight, 1));
	Util::printDeviceData(d_weight, out_dim.rows, in_dim.rows, 1, 1, "weight:");
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(in_dim.rows*out_dim.rows),
			&delta_scale, d_delta_weight, 1, d_weight, 1));
	Util::printDeviceData(d_weight, out_dim.rows, in_dim.rows, 1, 1, "weight:");

	Util::printDeviceData(d_delta_bias, out_dim.rows, 1, 1, 1, "delta_bias:");
	Util::printDeviceData(d_bias, out_dim.rows, 1, 1, 1, "bias:");
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(out_dim.rows),
			&b_delta_scale, d_delta_bias, 1, d_bias, 1));
	Util::printDeviceData(d_bias, out_dim.rows, 1, 1, 1, "bias:");
	*/

	int weight_size = this->in_dim.rows*this->out_dim.rows;
	Dtype norm_scale = 1.0/this->in_dim.batches;
	Dtype reg_scale = this->networkConfig->_weightDecay * weight_update_param.decay_mult;
	Dtype momentum = this->networkConfig->_momentum;
	Dtype learning_scale = this->networkConfig->_baseLearningRate * weight_update_param.lr_mult;
	Dtype negative_one = -1.0;

	_params[ParamType::Weight]->print_grad("weightGrad:");
	_params[ParamType::Weight]->print_data("weightData:");
	_paramsHistory[ParamType::Weight]->print_grad("weightHistoryGrad:");

	Dtype* d_weightGrad = _params[ParamType::Weight]->mutable_device_grad();
	Dtype* d_weightData = _params[ParamType::Weight]->mutable_device_data();
	Dtype* d_weightHistoryGrad = _paramsHistory[ParamType::Weight]->mutable_device_grad();

	checkCudaErrors(cublasSscal(Cuda::cublasHandle, static_cast<int>(weight_size), &norm_scale, d_weightGrad, 1));								// normalize by batch size
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(weight_size), &reg_scale, d_weightData, 1, d_weightGrad, 1));					// regularize
	checkCudaErrors(cublasSscal(Cuda::cublasHandle, static_cast<int>(weight_size), &momentum, d_weightHistoryGrad, 1));								//
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(weight_size), &learning_scale, d_weightGrad, 1, d_weightHistoryGrad, 1));	// momentum
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(weight_size), &negative_one, d_weightHistoryGrad, 1, d_weightData, 1));			// update

	_params[ParamType::Weight]->print_grad("weightGrad:");
	_params[ParamType::Weight]->print_data("weightData:");
	_paramsHistory[ParamType::Weight]->print_grad("weightHistoryGrad:");


	int bias_size = this->out_dim.rows;
	Dtype reg_scale_b = this->networkConfig->_weightDecay * bias_update_param.decay_mult;
	Dtype learning_scale_b = this->networkConfig->_baseLearningRate * bias_update_param.lr_mult;

	Dtype* d_biasGrad = _params[Bias]->mutable_device_grad();
	Dtype* d_biasData = _params[Bias]->mutable_device_data();
	Dtype* d_biasHistoryGrad = _paramsHistory[Bias]->mutable_device_grad();

	checkCudaErrors(cublasSscal(Cuda::cublasHandle, static_cast<int>(bias_size), &norm_scale, d_biasGrad, 1));								// normalize by batch size
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(bias_size), &reg_scale_b, d_biasData, 1, d_biasGrad, 1));					// regularize
	checkCudaErrors(cublasSscal(Cuda::cublasHandle, static_cast<int>(bias_size), &momentum, d_biasHistoryGrad, 1));								//
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(bias_size), &learning_scale_b, d_biasGrad, 1, d_biasHistoryGrad, 1));	// momentum
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(bias_size), &negative_one, d_biasHistoryGrad, 1, d_biasData, 1));				// update
}

template <typename Dtype>
void FullyConnectedLayer<Dtype>::_feedforward() {
	_params[Weight]->print_data("weightData:");
	this->_input->print_data("inputData:");

	// Apply weight to input data
	const Dtype* d_weightData = _params[Weight]->device_data();
	const Dtype* d_inputData = this->_input->device_data();
	Dtype* d_preActivationData = _preActivation->mutable_device_data();

	checkCudaErrors(cublasSgemm(Cuda::cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
			this->out_dim.rows, this->out_dim.batches, this->in_dim.rows,
			&Cuda::alpha, d_weightData, this->out_dim.rows, d_inputData, this->in_dim.rows,
			&Cuda::beta, d_preActivationData, this->out_dim.rows));

	_preActivation->print_data("preActivationData:");
	_params[Bias]->print_data("biasData:");

	// Add bias to weighted input data
	const Dtype* d_biasData = _params[Bias]->device_data();

	checkCudaErrors(cublasSgemm(Cuda::cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
			this->out_dim.rows, this->out_dim.batches, 1,
	    &Cuda::alpha,
	    d_biasData, this->out_dim.rows,
	    d_onevec, 1,
	    &Cuda::alpha,
	    d_preActivationData, this->out_dim.rows));

	// Activate weighted sum (+ bias)
	Dtype* d_outputData = this->_output->mutable_device_data();
	activation_fn->forward(this->outputTensorDesc, d_preActivationData, d_outputData);

	_preActivation->print_data(this->name+string("/d_preActivationData:"));
	this->_output->print_data(this->name+string("/d_outputData:"));


	/*
	// TODO skip when test
	if(Util::train && p_dropout < 1.0f) {
		int b_out = out_dim.batchsize();
		for(int i = 0; i < b_out; i++) {
			mask[i] = ((rand()/(RAND_MAX+1.0) > p_dropout)?1:0);
		}
		checkCudaErrors(cudaMemcpyAsync(d_mask, mask, sizeof(Dtype)*b_out, cudaMemcpyHostToDevice));
		//FillOnes<<<RoundUp(in_dim.batches, BW), BW>>>(this->d_onevec, in_dim.batches);
		Dropout<<<RoundUp(b_out, BW), BW>>>(b_out, d_output, d_mask, 0, scale, d_output);

		//Util::setPrint(true);
		Util::printData(mask, out_dim.rows, out_dim.batches, 1, 1, this->name+string("/mask:"));
		Util::printDeviceData(d_output, out_dim.rows, out_dim.batches, 1, 1, this->name+string("/d_output:"));
		//Util::setPrint(false);
	}
	*/
}



template <typename Dtype>
void FullyConnectedLayer<Dtype>::_backpropagation() {
	/*
	if(Util::train && p_dropout < 1.0f) {
		//Util::setPrint(true);
		Util::printDeviceData(d_delta_output, out_dim.rows, out_dim.batches, 1, 1, "delta_input:");
		Dropout<<<RoundUp(out_dim.batchsize(), BW), BW>>>(out_dim.batchsize(), d_delta_output, d_mask, 0, scale, d_delta_output);

		Util::printData(mask, out_dim.rows, out_dim.batches, 1, 1, this->name+string("/mask:"));
		Util::printDeviceData(d_delta_output, out_dim.rows, out_dim.batches, 1, 1, "delta_output:");
		Util::setPrint(false);
	}
	*/

	/*
	this->_output->print_data("output:");

	const Dtype* d_output = this->_output->device_data();
	const Dtype* d_delta_output = this->_output->device_grad();
	const Dtype* d_z = this->_preActivation->device_data();
	Dtype* d_delta = this->_preActivation->mutable_device_grad();

	this->activation_fn->backward(this->outputTensorDesc, d_output, d_delta_output, d_z, d_delta);
	*/

	/*
	if(this->_preActivation->is_nan_grad()) {
		cout << this->name << " _preActivation gradient nan ... " << endl;

		Data<Dtype>::printConfig = 1;
		this->_output->print_data("output:");
		this->_output->print_grad("deltaOutput:");
		this->_preActivation->print_data("preActivation:");
		this->_preActivation->print_grad("delta:");
		Data<Dtype>::printConfig = 0;
		exit(1);
	}
	*/

	_computePreActivationGrad();
	_computeWeightGrad();
	_computeBiasGrad();
	_computeInputGrad();

}


template <typename Dtype>
void FullyConnectedLayer<Dtype>::_computePreActivationGrad() {
	this->_output->print_data("output:");

	const Dtype* d_y = this->_output->device_data();
	const Dtype* d_dy = this->_output->device_grad();
	const Dtype* d_x = this->_preActivation->device_data();
	Dtype* d_dx = this->_preActivation->mutable_device_grad();

	this->activation_fn->backward(this->outputTensorDesc, d_y, d_dy, d_x, d_dx);
}



template <typename Dtype>
void FullyConnectedLayer<Dtype>::_computeWeightGrad() {
	// d(Cost)/d(Weight)
	const Dtype* d_preActivationGrad = this->_preActivation->device_grad();
	const Dtype* d_inputData = this->_input->device_data();
	Dtype* d_weightGrad = this->_params[Weight]->mutable_device_grad();

	checkCudaErrors(cublasSgemm(Cuda::cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
			this->out_dim.rows, this->in_dim.rows, this->out_dim.batches,
			&Cuda::alpha, d_preActivationGrad, this->out_dim.rows, d_inputData, this->in_dim.rows,
			&Cuda::beta, d_weightGrad, this->out_dim.rows));

	/*
	Data<Dtype>::printConfig = 1;
	this->_preActivation->print_grad("preActivationGrad:");
	this->_input->print_data("inputData:");
	this->_params[Weight]->print_grad("weightGrad");
	Data<Dtype>::printConfig = 0;
	*/

	/*
	if(this->_params[Weight]->is_nan_grad()) {
		cout << this->name << " _params weight gradient nan ... " << endl;
		Data<Dtype>::printConfig = 1;
		this->_params[Weight]->print_grad("deltaWeight:");
		Data<Dtype>::printConfig = 0;
		exit(1);
	}
	*/
}

template <typename Dtype>
void FullyConnectedLayer<Dtype>::_computeBiasGrad() {
	// d(Cost)/d(Bias) (same as d_preActivationGrad)
	const Dtype* d_preActivationGrad = this->_preActivation->device_grad();
	Dtype* d_biasGrad = _params[Bias]->mutable_device_grad();

	checkCudaErrors(cublasSgemv(Cuda::cublasHandle, CUBLAS_OP_N,
			this->out_dim.rows, this->out_dim.batches,
			&Cuda::alpha, d_preActivationGrad, this->out_dim.rows, d_onevec, 1,
			&Cuda::beta, d_biasGrad, 1));
	_params[Bias]->print_grad("biasGrad:");
	_params[Weight]->print_data("weightData:");
	_preActivation->print_grad("preActivationGrad");
}

template <typename Dtype>
void FullyConnectedLayer<Dtype>::_computeInputGrad() {
	// d(Cost)/d(Input)
	const Dtype* d_weightData = _params[Weight]->device_data();
	const Dtype* d_preActivationGrad = this->_preActivation->device_grad();
	Dtype* d_inputGrad = this->_input->mutable_device_grad();

	checkCudaErrors(cublasSgemm(Cuda::cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
			this->in_dim.rows, this->out_dim.batches, this->out_dim.rows,
			&Cuda::alpha, d_weightData, this->out_dim.rows, d_preActivationGrad, this->out_dim.rows,
			&Cuda::beta, d_inputGrad, this->in_dim.rows));
	this->_input->print_grad("inputGrad:");

	/*
	if(this->_input->is_nan_grad()) {
		cout << this->name << " _input gradient nan ... " << endl;
		Data<Dtype>::printConfig = 1;
		this->_input->print_grad("deltaInput:");
		Data<Dtype>::printConfig = 0;
		exit(1);
	}
	*/
}




template FullyConnectedLayer<float>::~FullyConnectedLayer();
template void FullyConnectedLayer<float>::_shape(bool recursive);
template void FullyConnectedLayer<float>::_clearShape();
template void FullyConnectedLayer<float>::update();
template void FullyConnectedLayer<float>::_feedforward();
template void FullyConnectedLayer<float>::_backpropagation();



#endif








