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





FullyConnectedLayer::~FullyConnectedLayer() {
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

	ActivationFactory::destory(activation_fn);
}



void FullyConnectedLayer::_shape(bool recursive) {
	in_dim.rows = in_dim.rows*in_dim.cols*in_dim.channels;
	in_dim.cols = 1;
	in_dim.channels = 1;
	out_dim.batches = in_dim.batches;

	if(recursive) {
		HiddenLayer::_shape();
	}

	uint32_t u_in = in_dim.unitsize();
	uint32_t u_out = out_dim.unitsize();
	uint32_t b_in = in_dim.batchsize();
	uint32_t b_out = out_dim.batchsize();

	//weight = new DATATYPE[u_out*u_in];
	//bias = new DATATYPE[u_out];
	_params[ParamType::Weight]->reshape({1, 1, u_out*u_in, 1});
	_params[ParamType::Bias]->reshape({1, 1, u_out, 1});
	_paramsHistory[ParamType::Weight]->reshape({1, 1, u_out*u_in, 1});
	_paramsHistory[ParamType::Bias]->reshape({1, 1, u_out, 1});
	_preActivation->reshape({out_dim.batches, 1, u_out, 1});


	//cout << this->name << ", fanin: " << u_out*u_in << endl;
	weight_filler.fill(_params[ParamType::Weight]->mutable_host_data(), u_out*u_in, u_in, u_out);
	bias_filler.fill(_params[ParamType::Bias]->mutable_host_data(), u_out, u_in, u_out);

	//Util::printData(weight, u_out, u_in, 1, 1, "weight:");
	//Util::printData(bias, u_out, 1, 1, 1, "bias:");
	_params[ParamType::Weight]->print_data("weight:");
	_params[ParamType::Bias]->print_data("bias:");


	//checkCudaErrors(Util::ucudaMalloc(&this->d_weight, sizeof(DATATYPE)*u_out*u_in));
	//checkCudaErrors(Util::ucudaMalloc(&this->d_bias, sizeof(DATATYPE)*u_out));

	//checkCudaErrors(Util::ucudaMalloc(&this->d_z, sizeof(DATATYPE)*b_out));
	//checkCudaErrors(Util::ucudaMalloc(&this->d_delta, sizeof(DATATYPE)*b_out));
	//checkCudaErrors(Util::ucudaMalloc(&this->d_delta_input, sizeof(DATATYPE)*b_in));

	//checkCudaErrors(Util::ucudaMalloc(&this->d_delta_weight, sizeof(DATATYPE)*u_out*u_in));
	//checkCudaErrors(Util::ucudaMalloc(&this->d_delta_weight_prev, sizeof(DATATYPE)*u_out*u_in));
	//FillValues<<<RoundUp(u_out*u_in, BW), BW>>>(this->d_onevec, u_out*u_in, 0.0f);
	//checkCudaErrors(cudaMemset(d_delta_weight_prev, 0, u_out*u_in*sizeof(DATATYPE)));

	//checkCudaErrors(Util::ucudaMalloc(&this->d_delta_bias, sizeof(DATATYPE)*u_out));
	//checkCudaErrors(Util::ucudaMalloc(&this->d_delta_bias_prev, sizeof(DATATYPE)*u_out));
	//checkCudaErrors(cudaMemset(d_delta_bias_prev, 0, u_out*sizeof(DATATYPE)));

	checkCudaErrors(Util::ucudaMalloc(&this->d_onevec, sizeof(DATATYPE)*in_dim.batches));
	FillValues<<<RoundUp(in_dim.batches, BW), BW>>>(this->d_onevec, in_dim.batches, 1.0f);
	//checkCudaErrors(cudaMemset(d_onevec, 1, in_dim.batches));


	//checkCudaErrors(cudaMemcpyAsync(this->d_weight, weight, sizeof(DATATYPE)*u_out*u_in, cudaMemcpyHostToDevice));
	//checkCudaErrors(cudaMemcpyAsync(this->d_bias, bias, sizeof(DATATYPE)*u_out, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaDeviceSynchronize());

	mask = new DATATYPE[b_out];
	checkCudaErrors(Util::ucudaMalloc(&this->d_mask, sizeof(DATATYPE)*b_out));
}

void FullyConnectedLayer::_clearShape() {
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

	HiddenLayer::_clearShape();
}


//void FullyConnectedLayer::_update(UINT n, UINT miniBatchSize) {
void FullyConnectedLayer::update() {
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

	int weight_size = in_dim.rows*out_dim.rows;
	DATATYPE norm_scale = 1.0/in_dim.batches;
	DATATYPE reg_scale = networkConfig->_weightDecay * weight_update_param.decay_mult;
	DATATYPE momentum = networkConfig->_momentum;
	DATATYPE learning_scale = networkConfig->_baseLearningRate * weight_update_param.lr_mult;
	DATATYPE negative_one = -1.0;

	//Util::setPrint(true);
	//Util::printDeviceData(d_delta_weight, out_dim.rows, in_dim.rows, 1, 1, "d_delta_weight:");
	//Util::printDeviceData(d_weight, out_dim.rows, in_dim.rows, 1, 1, "d_weight:");
	//Util::printDeviceData(d_delta_weight_prev, out_dim.rows, in_dim.rows, 1, 1, "d_delta_weight_prev:");
	_params[ParamType::Weight]->print_grad("d_delta_weight:");
	_params[ParamType::Weight]->print_data("d_weight:");
	_paramsHistory[ParamType::Weight]->print_grad("d_delta_weight_prev:");

	DATATYPE* d_delta_weight = _params[ParamType::Weight]->mutable_device_grad();
	DATATYPE* d_weight = _params[ParamType::Weight]->mutable_device_data();
	DATATYPE* d_delta_weight_prev = _paramsHistory[ParamType::Weight]->mutable_device_grad();

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


	int bias_size = out_dim.rows;
	DATATYPE reg_scale_b = networkConfig->_weightDecay * bias_update_param.decay_mult;
	DATATYPE learning_scale_b = networkConfig->_baseLearningRate * bias_update_param.lr_mult;

	DATATYPE* d_delta_bias = _params[Bias]->mutable_device_grad();
	DATATYPE* d_bias = _params[Bias]->mutable_device_data();
	DATATYPE* d_delta_bias_prev = _paramsHistory[Bias]->mutable_device_grad();

	checkCudaErrors(cublasSscal(Cuda::cublasHandle, static_cast<int>(bias_size), &norm_scale, d_delta_bias, 1));								// normalize by batch size
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(bias_size), &reg_scale_b, d_bias, 1, d_delta_bias, 1));					// regularize
	checkCudaErrors(cublasSscal(Cuda::cublasHandle, static_cast<int>(bias_size), &momentum, d_delta_bias_prev, 1));								//
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(bias_size), &learning_scale_b, d_delta_bias, 1, d_delta_bias_prev, 1));	// momentum
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(bias_size), &negative_one, d_delta_bias_prev, 1, d_bias, 1));				// update
}

void FullyConnectedLayer::_feedforward() {
	//Util::printDeviceData(d_weight, out_dim.rows, in_dim.rows, 1, 1, "d_weight:");
	//Util::printDeviceData(d_input, in_dim.rows, in_dim.batches, 1, 1, "d_input:");
	_params[Weight]->print_data("d_weight:");
	_input->print_data("d_input:");

	const DATATYPE* d_weight = _params[Weight]->device_data();
	const DATATYPE* d_input = _input->device_data();
	DATATYPE* d_z = _preActivation->mutable_device_data();

	checkCudaErrors(cublasSgemm(Cuda::cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
			out_dim.rows, out_dim.batches, in_dim.rows,
			&Cuda::alpha,
			d_weight, out_dim.rows,
			d_input, in_dim.rows,
			&Cuda::beta,
			d_z, out_dim.rows));

	//Util::printDeviceData(d_z, out_dim.rows, out_dim.batches, 1, 1, "d_z:");
	//Util::printDeviceData(d_bias, out_dim.rows, 1, 1, 1, "d_b:");
	_preActivation->print_data("d_z:");
	_params[Bias]->print_data("d_b:");

	const DATATYPE* d_bias = _params[Bias]->device_data();

	checkCudaErrors(cublasSgemm(Cuda::cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
			out_dim.rows, out_dim.batches, 1,
	    &Cuda::alpha,
	    d_bias, out_dim.rows,
	    d_onevec, 1,
	    &Cuda::alpha,
	    d_z, out_dim.rows));

	//Util::printDeviceData(d_z, out_dim.rows, out_dim.batches, 1, 1, "d_z:");
	//Util::printDeviceData(d_output, out_dim.rows, out_dim.cols, out_dim.channels, out_dim.batches, "d_output:");
	_preActivation->print_data("d_z:");

	DATATYPE* d_output = _output->mutable_device_data();

	activation_fn->activate(d_z, d_output, outputTensorDesc);

	//Util::printDeviceData(d_z, out_dim.rows, out_dim.batches, 1, 1, this->name+string("/d_z:"));
	//Util::printDeviceData(d_output, out_dim.rows, out_dim.batches, 1, 1, this->name+string("/d_output:"));
	_preActivation->print_data(this->name+string("/d_z:"));
	_output->print_data(this->name+string("/d_output:"));

	//exit(1);


	/*
	// TODO skip when test
	if(Util::train && p_dropout < 1.0f) {
		int b_out = out_dim.batchsize();
		for(int i = 0; i < b_out; i++) {
			mask[i] = ((rand()/(RAND_MAX+1.0) > p_dropout)?1:0);
		}
		checkCudaErrors(cudaMemcpyAsync(d_mask, mask, sizeof(DATATYPE)*b_out, cudaMemcpyHostToDevice));
		//FillOnes<<<RoundUp(in_dim.batches, BW), BW>>>(this->d_onevec, in_dim.batches);
		Dropout<<<RoundUp(b_out, BW), BW>>>(b_out, d_output, d_mask, 0, scale, d_output);

		//Util::setPrint(true);
		Util::printData(mask, out_dim.rows, out_dim.batches, 1, 1, this->name+string("/mask:"));
		Util::printDeviceData(d_output, out_dim.rows, out_dim.batches, 1, 1, this->name+string("/d_output:"));
		//Util::setPrint(false);
	}
	*/
}

void FullyConnectedLayer::_backpropagation() {
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
	_output->print_data("output:");

	const DATATYPE* d_output = _output->device_data();
	const DATATYPE* d_delta_output = _output->device_grad();
	const DATATYPE* d_z = _preActivation->device_data();
	DATATYPE* d_delta = _preActivation->mutable_device_grad();

	activation_fn->d_activate(d_output, d_delta_output, d_z, d_delta, outputTensorDesc);

	//Util::printDeviceData(d_delta, out_dim.rows, out_dim.batches, 1, 1, "d_delta:");
	_preActivation->print_grad("d_delta:");

	//Util::printDeviceData(d_input, in_dim.rows, in_dim.batches, 1, 1, "d_input:");
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
	//checkCudaErrors(cublasSgemm(Cuda::cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, in_dim.rows, out_dim.batches, out_dim.rows,
			//&Cuda::alpha, d_weight, out_dim.rows, d_delta, out_dim.rows, &Cuda::beta, d_delta_input, in_dim.rows));

	uint32_t status = cublasSgemm(Cuda::cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, in_dim.rows, out_dim.batches, out_dim.rows,
				&Cuda::alpha, d_weight, out_dim.rows, d_delta, out_dim.rows, &Cuda::beta, d_delta_input, in_dim.rows);
	std::stringstream _error;
	if (status != 0) {
	  _error << "Cuda failure: " << status;
	  FatalError(_error.str());
	}


	//Util::printDeviceData(d_delta_input, in_dim.rows, in_dim.batches, 1, 1, "d_delta_input:");
	_input->print_grad("d_delta_input:");

	/*
	 * rcube w_next_delta(size(output));
	rcube sp;
	activation_fn->d_activate(output, sp);

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







#endif








