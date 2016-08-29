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
__global__ void FillValues(DATATYPE *vec, int size, DATATYPE value)
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
__global__ void Dropout(const int n, const DATATYPE* in, const DATATYPE* mask,
		const unsigned int threashold, const float scale, DATATYPE *out)
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

	//if(weight) delete [] weight;
	//if(bias) delete [] bias;

	//checkCudaErrors(cudaFree(d_weight));
	//checkCudaErrors(cudaFree(d_bias));

	//checkCudaErrors(cudaFree(d_z));
	//checkCudaErrors(cudaFree(d_delta));
	//checkCudaErrors(cudaFree(d_delta_input));
	//checkCudaErrors(cudaFree(d_delta_weight));
	//checkCudaErrors(cudaFree(d_delta_weight_prev));
	//checkCudaErrors(cudaFree(d_delta_bias));
	//checkCudaErrors(cudaFree(d_delta_bias_prev));

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
	_params[ParamType::Weight]->reshape({1, 1, u_out*u_in, 1});
	_params[ParamType::Bias]->reshape({1, 1, u_out, 1});
	_paramsHistory[ParamType::Weight]->reshape({1, 1, u_out*u_in, 1});
	_paramsHistory[ParamType::Bias]->reshape({1, 1, u_out, 1});
	_preActivation->reshape({this->out_dim.batches, 1, u_out, 1});


	//cout << this->name << ", fanin: " << u_out*u_in << endl;
	weight_filler.fill(_params[ParamType::Weight]->mutable_host_data(), u_out*u_in, u_in, u_out);
	bias_filler.fill(_params[ParamType::Bias]->mutable_host_data(), u_out, u_in, u_out);

	//Util::printData(weight, u_out, u_in, 1, 1, "weight:");
	//Util::printData(bias, u_out, 1, 1, 1, "bias:");
	_params[ParamType::Weight]->print_data("weight:");
	_params[ParamType::Bias]->print_data("bias:");


	//checkCudaErrors(Util::ucudaMalloc(&this->d_weight, sizeof(Dtype)*u_out*u_in));
	//checkCudaErrors(Util::ucudaMalloc(&this->d_bias, sizeof(Dtype)*u_out));

	//checkCudaErrors(Util::ucudaMalloc(&this->d_z, sizeof(Dtype)*b_out));
	//checkCudaErrors(Util::ucudaMalloc(&this->d_delta, sizeof(Dtype)*b_out));
	//checkCudaErrors(Util::ucudaMalloc(&this->d_delta_input, sizeof(Dtype)*b_in));

	//checkCudaErrors(Util::ucudaMalloc(&this->d_delta_weight, sizeof(Dtype)*u_out*u_in));
	//checkCudaErrors(Util::ucudaMalloc(&this->d_delta_weight_prev, sizeof(Dtype)*u_out*u_in));
	//FillValues<<<RoundUp(u_out*u_in, BW), BW>>>(this->d_onevec, u_out*u_in, 0.0f);
	//checkCudaErrors(cudaMemset(d_delta_weight_prev, 0, u_out*u_in*sizeof(Dtype)));

	//checkCudaErrors(Util::ucudaMalloc(&this->d_delta_bias, sizeof(Dtype)*u_out));
	//checkCudaErrors(Util::ucudaMalloc(&this->d_delta_bias_prev, sizeof(Dtype)*u_out));
	//checkCudaErrors(cudaMemset(d_delta_bias_prev, 0, u_out*sizeof(Dtype)));

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
	//if(weight) delete [] weight;
	//if(bias) delete [] bias;

	//checkCudaErrors(cudaFree(d_weight));
	//checkCudaErrors(cudaFree(d_bias));

	//checkCudaErrors(cudaFree(d_z));
	//checkCudaErrors(cudaFree(d_delta));
	//checkCudaErrors(cudaFree(d_delta_input));
	//checkCudaErrors(cudaFree(d_delta_weight));
	//checkCudaErrors(cudaFree(d_delta_weight_prev));
	//checkCudaErrors(cudaFree(d_delta_bias));
	//checkCudaErrors(cudaFree(d_delta_bias_prev));

	delete _params[0];
	delete _params[1];
	//_params.clear();

	delete _paramsHistory[0];
	delete _paramsHistory[1];
	//_paramsHistory.clear();

	delete _preActivation;


	if(mask) delete [] mask;
	checkCudaErrors(cudaFree(d_mask));

	//weight = NULL;
	//bias = NULL;

	//d_weight = NULL;
	//d_bias = NULL;

	//d_z = NULL;
	//d_delta = NULL;

	//d_delta_weight = NULL;
	//d_delta_weight_prev = NULL;
	//d_delta_bias = NULL;
	//d_delta_bias_prev = NULL;

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

	Util::printDeviceData(d_delta_weight, out_dim.rows, in_dim.rows, 1, 1, "d_delta_weight:");
	Util::printDeviceData(d_weight, out_dim.rows, in_dim.rows, 1, 1, "d_weight:");
	checkCudaErrors(cublasSscal(Cuda::cublasHandle, static_cast<int>(in_dim.rows*out_dim.rows), &param_scale, d_weight, 1));
	Util::printDeviceData(d_weight, out_dim.rows, in_dim.rows, 1, 1, "d_weight:");
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(in_dim.rows*out_dim.rows),
			&delta_scale, d_delta_weight, 1, d_weight, 1));
	Util::printDeviceData(d_weight, out_dim.rows, in_dim.rows, 1, 1, "d_weight:");

	Util::printDeviceData(d_delta_bias, out_dim.rows, 1, 1, 1, "d_delta_bias:");
	Util::printDeviceData(d_bias, out_dim.rows, 1, 1, 1, "d_bias:");
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(out_dim.rows),
			&b_delta_scale, d_delta_bias, 1, d_bias, 1));
	Util::printDeviceData(d_bias, out_dim.rows, 1, 1, 1, "d_bias:");
	*/

	int weight_size = this->in_dim.rows*this->out_dim.rows;
	Dtype norm_scale = 1.0/this->in_dim.batches;
	Dtype reg_scale = this->networkConfig->_weightDecay * weight_update_param.decay_mult;
	Dtype momentum = this->networkConfig->_momentum;
	Dtype learning_scale = this->networkConfig->_baseLearningRate * weight_update_param.lr_mult;
	Dtype negative_one = -1.0;

	//Util::setPrint(true);
	//Util::printDeviceData(d_delta_weight, out_dim.rows, in_dim.rows, 1, 1, "d_delta_weight:");
	//Util::printDeviceData(d_weight, out_dim.rows, in_dim.rows, 1, 1, "d_weight:");
	//Util::printDeviceData(d_delta_weight_prev, out_dim.rows, in_dim.rows, 1, 1, "d_delta_weight_prev:");
	_params[ParamType::Weight]->print_grad("d_delta_weight:");
	_params[ParamType::Weight]->print_data("d_weight:");
	_paramsHistory[ParamType::Weight]->print_grad("d_delta_weight_prev:");

	Dtype* d_delta_weight = _params[ParamType::Weight]->mutable_device_grad();
	Dtype* d_weight = _params[ParamType::Weight]->mutable_device_data();
	Dtype* d_delta_weight_prev = _paramsHistory[ParamType::Weight]->mutable_device_grad();

	checkCudaErrors(cublasSscal(Cuda::cublasHandle, static_cast<int>(weight_size), &norm_scale, d_delta_weight, 1));								// normalize by batch size
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(weight_size), &reg_scale, d_weight, 1, d_delta_weight, 1));					// regularize
	checkCudaErrors(cublasSscal(Cuda::cublasHandle, static_cast<int>(weight_size), &momentum, d_delta_weight_prev, 1));								//
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(weight_size), &learning_scale, d_delta_weight, 1, d_delta_weight_prev, 1));	// momentum
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(weight_size), &negative_one, d_delta_weight_prev, 1, d_weight, 1));			// update

	//Util::printDeviceData(d_delta_weight, out_dim.rows, in_dim.rows, 1, 1, "d_delta_weight:");
	//Util::printDeviceData(d_weight, out_dim.rows, in_dim.rows, 1, 1, "d_weight:");
	//Util::printDeviceData(d_delta_weight_prev, out_dim.rows, in_dim.rows, 1, 1, "d_delta_weight_prev:");
	_params[ParamType::Weight]->print_grad("d_delta_weight:");
	_params[ParamType::Weight]->print_data("d_weight:");
	_paramsHistory[ParamType::Weight]->print_grad("d_delta_weight_prev:");


	int bias_size = this->out_dim.rows;
	Dtype reg_scale_b = this->networkConfig->_weightDecay * bias_update_param.decay_mult;
	Dtype learning_scale_b = this->networkConfig->_baseLearningRate * bias_update_param.lr_mult;

	Dtype* d_delta_bias = _params[Bias]->mutable_device_grad();
	Dtype* d_bias = _params[Bias]->mutable_device_data();
	Dtype* d_delta_bias_prev = _paramsHistory[Bias]->mutable_device_grad();

	checkCudaErrors(cublasSscal(Cuda::cublasHandle, static_cast<int>(bias_size), &norm_scale, d_delta_bias, 1));								// normalize by batch size
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(bias_size), &reg_scale_b, d_bias, 1, d_delta_bias, 1));					// regularize
	checkCudaErrors(cublasSscal(Cuda::cublasHandle, static_cast<int>(bias_size), &momentum, d_delta_bias_prev, 1));								//
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(bias_size), &learning_scale_b, d_delta_bias, 1, d_delta_bias_prev, 1));	// momentum
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(bias_size), &negative_one, d_delta_bias_prev, 1, d_bias, 1));				// update
}

template <typename Dtype>
void FullyConnectedLayer<Dtype>::_feedforward() {
	//Util::printDeviceData(d_weight, out_dim.rows, in_dim.rows, 1, 1, "d_weight:");
	//Util::printDeviceData(d_input, in_dim.rows, in_dim.batches, 1, 1, "d_input:");
	_params[Weight]->print_data("d_weight:");
	this->_input->print_data("d_input:");

	const Dtype* d_weight = _params[Weight]->device_data();
	const Dtype* d_input = this->_input->device_data();
	Dtype* d_z = _preActivation->mutable_device_data();

	checkCudaErrors(cublasSgemm(Cuda::cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
			this->out_dim.rows, this->out_dim.batches, this->in_dim.rows,
			&Cuda::alpha,
			d_weight, this->out_dim.rows,
			d_input, this->in_dim.rows,
			&Cuda::beta,
			d_z, this->out_dim.rows));

	//Util::printDeviceData(d_z, out_dim.rows, out_dim.batches, 1, 1, "d_z:");
	//Util::printDeviceData(d_bias, out_dim.rows, 1, 1, 1, "d_b:");
	_preActivation->print_data("d_z:");
	_params[Bias]->print_data("d_b:");

	const Dtype* d_bias = _params[Bias]->device_data();

	checkCudaErrors(cublasSgemm(Cuda::cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
			this->out_dim.rows, this->out_dim.batches, 1,
	    &Cuda::alpha,
	    d_bias, this->out_dim.rows,
	    d_onevec, 1,
	    &Cuda::alpha,
	    d_z, this->out_dim.rows));

	Dtype* d_output = this->_output->mutable_device_data();
	//const Dtype* h_output = _output->host_data();
	//const Dtype* h_z = _preActivation->host_data();

	activation_fn->forward(this->outputTensorDesc, d_z, d_output);

	//Util::printDeviceData(d_z, out_dim.rows, out_dim.batches, 1, 1, this->name+string("/d_z:"));
	//Util::printDeviceData(d_output, out_dim.rows, out_dim.batches, 1, 1, this->name+string("/d_output:"));
	_preActivation->print_data(this->name+string("/d_z:"));
	this->_output->print_data(this->name+string("/d_output:"));




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
	Cuda::refresh();
	/*
	if(Util::train && p_dropout < 1.0f) {
		//Util::setPrint(true);
		Util::printDeviceData(d_delta_output, out_dim.rows, out_dim.batches, 1, 1, "delta_input:");
		Dropout<<<RoundUp(out_dim.batchsize(), BW), BW>>>(out_dim.batchsize(), d_delta_output, d_mask, 0, scale, d_delta_output);

		Util::printData(mask, out_dim.rows, out_dim.batches, 1, 1, this->name+string("/mask:"));
		Util::printDeviceData(d_delta_output, out_dim.rows, out_dim.batches, 1, 1, "d_delta_output:");
		Util::setPrint(false);
	}
	*/
	//Util::printDeviceData(d_output, out_dim.rows, out_dim.batches, 1, 1, "output:");
	this->_output->print_data("output:");

	const Dtype* d_output = this->_output->device_data();
	const Dtype* d_delta_output = this->_output->device_grad();
	const Dtype* d_z = this->_preActivation->device_data();
	Dtype* d_delta = this->_preActivation->mutable_device_grad();

	//activation_fn->backward(d_output, d_delta_output, d_z, d_delta, outputTensorDesc);
	this->activation_fn->backward(this->outputTensorDesc, d_output, d_delta_output, d_z, d_delta);

	//Util::printDeviceData(d_delta, out_dim.rows, out_dim.batches, 1, 1, "d_delta:");
	this->_preActivation->print_grad("d_delta:");

	//Util::printDeviceData(d_input, in_dim.rows, in_dim.batches, 1, 1, "d_input:");
	this->_input->print_data("d_input:");
	const Dtype* d_input = this->_input->device_data();
	Dtype* d_delta_weight = this->_params[Weight]->mutable_device_grad();

	checkCudaErrors(cublasSgemm(Cuda::cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, this->out_dim.rows, this->in_dim.rows, this->out_dim.batches,
			&Cuda::alpha, d_delta, this->out_dim.rows, d_input, this->in_dim.rows, &Cuda::beta, d_delta_weight, this->out_dim.rows));
	//Util::printDeviceData(d_delta_weight, out_dim.rows, in_dim.rows, 1, 1, "d_delta_weight:");
	_params[Weight]->print_grad("d_delta_weight:");

	Dtype* d_delta_bias = _params[Bias]->mutable_device_grad();
	checkCudaErrors(cublasSgemv(Cuda::cublasHandle, CUBLAS_OP_N, this->out_dim.rows, this->out_dim.batches,
			&Cuda::alpha, d_delta, this->out_dim.rows, d_onevec, 1, &Cuda::beta, d_delta_bias, 1));
	//Util::printDeviceData(d_delta_bias, out_dim.rows, 1, 1, 1, "d_delta_bias:");
	_params[Bias]->print_grad("d_delta_bias:");

	//Util::printDeviceData(d_weight, out_dim.rows, in_dim.rows, 1, 1, "d_weight:");
	//Util::printDeviceData(d_delta, out_dim.rows, out_dim.batches, 1, 1, "d_delta:");
	_params[Weight]->print_data("d_weight:");
	_preActivation->print_grad("d_delta");

	const Dtype* d_weight = _params[Weight]->device_data();
	Dtype* d_delta_input = this->_input->mutable_device_grad();
	//checkCudaErrors(cublasSgemm(Cuda::cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, in_dim.rows, out_dim.batches, out_dim.rows,
			//&Cuda::alpha, d_weight, out_dim.rows, d_delta, out_dim.rows, &Cuda::beta, d_delta_input, in_dim.rows));

	uint32_t status = cublasSgemm(Cuda::cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, this->in_dim.rows, this->out_dim.batches, this->out_dim.rows,
				&Cuda::alpha, d_weight, this->out_dim.rows, d_delta, this->out_dim.rows, &Cuda::beta, d_delta_input, this->in_dim.rows);
	std::stringstream _error;
	if (status != 0) {
	  _error << "Cuda failure: " << status;
	  FatalError(_error.str());
	}


	//Util::printDeviceData(d_delta_input, in_dim.rows, in_dim.batches, 1, 1, "d_delta_input:");
	this->_input->print_grad("d_delta_input:");

	/*
	 * rcube w_next_delta(size(output));
	rcube sp;
	activation_fn->backward(output, sp);

	// delta l = dC/dz
	delta.slice(0) = w_next_delta.slice(0) % sp.slice(0);

	nabla_b += delta.slice(0);
	// delta lw = dC/dw
	nabla_w += delta.slice(0)*input.slice(0).t();

	// delta lx = dC/dx
	delta_input.slice(0) = weight.t()*delta.slice(0);
	//fc_layer->getWeight().t()*fc_layer->getDelta().slice(0)
	 */
}



template FullyConnectedLayer<float>::~FullyConnectedLayer();
template void FullyConnectedLayer<float>::_shape(bool recursive);
template void FullyConnectedLayer<float>::_clearShape();
template void FullyConnectedLayer<float>::update();
template void FullyConnectedLayer<float>::_feedforward();
template void FullyConnectedLayer<float>::_backpropagation();



#endif








