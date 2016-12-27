/*
 * FullyConnectedLayer.cpp
 *
 *  Created on: 2016. 5. 10.
 *      Author: jhkim
 */

#include "FullyConnectedLayer.h"
#include "Util.h"
#include "Exception.h"
#include "NetworkConfig.h"
#include "cuda_runtime.h"
#include <algorithm>

using namespace std;

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

/**
 * dst array에 src array를 더한다.
 *
 * @param dst dst array, dst + src가 저장이 될 장소
 * @param src src array
 * @param N The number of elements in the array.
 */
template <typename Dtype>
__global__ void AddArrayOfFCLayer(Dtype* dst, const Dtype* src, int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= N)
		return;

	dst[idx] = dst[idx] + src[idx];
}

template <typename Dtype>
FullyConnectedLayer<Dtype>::~FullyConnectedLayer() {
	//delete _params[ParamType::Weight];
	//delete _params[ParamType::Bias];
	//_params.clear();
	Util::clearVector(_params);

	//delete _paramsHistory[ParamType::Weight];
	//delete _paramsHistory[ParamType::Bias];
	//_paramsHistory.clear();
	Util::clearVector(_paramsHistory);

	delete _preActivation;
	checkCudaErrors(cudaFree(d_onevec));

	ActivationFactory<Dtype>::destory(activation_fn);
}


template <typename Dtype>
void FullyConnectedLayer<Dtype>::_shape(bool recursive) {
	this->setInDimension(this->_inputData[0]->getShape());

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
	//_params[ParamType::Weight]->reshape({1, 1, u_out, u_in});
	_params[ParamType::Weight]->shape({u_out, u_in, 1, 1});
	_params[ParamType::Bias]->shape({1, 1, u_out, 1});
	_paramsHistory[ParamType::Weight]->shape({u_out, u_in, 1, 1});
	_paramsHistory[ParamType::Bias]->shape({1, 1, u_out, 1});
	_preActivation->shape({this->out_dim.batches, 1, u_out, 1});


	//cout << this->name << ", fanin: " << u_out*u_in << endl;
	//weight_filler.fill(_params[ParamType::Weight]->mutable_host_data(), u_out*u_in, u_in, u_out);
	//bias_filler.fill(_params[ParamType::Bias]->mutable_host_data(), u_out, u_in, u_out);
	weight_filler.fill(_params[ParamType::Weight]);
	bias_filler.fill(_params[ParamType::Bias]);

	_params[ParamType::Weight]->print_data("weight:");
	_params[ParamType::Bias]->print_data("bias:");

	checkCudaErrors(Util::ucudaMalloc(&this->d_onevec, sizeof(Dtype)*this->in_dim.batches));
	FillValues<<<RoundUp(this->in_dim.batches, BW), BW>>>(this->d_onevec, this->in_dim.batches, 1.0f);
	//cuda_FillValues(this->d_onevec, in_dim.batches, 1.0f);
	//checkCudaErrors(cudaMemset(d_onevec, 1, in_dim.batches));


	//checkCudaErrors(cudaMemcpyAsync(this->d_weight, weight, sizeof(Dtype)*u_out*u_in, cudaMemcpyHostToDevice));
	//checkCudaErrors(cudaMemcpyAsync(this->d_bias, bias, sizeof(Dtype)*u_out, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaDeviceSynchronize());

	//mask = new Dtype[b_out];
	//checkCudaErrors(Util::ucudaMalloc(&this->d_mask, sizeof(Dtype)*b_out));

	_mask.shape(b_out);
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


	//if(mask) delete [] mask;
	//checkCudaErrors(cudaFree(d_mask));
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

	/*
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
	*/


	const uint32_t weightSize = this->in_dim.rows*this->out_dim.rows;
	const Dtype regScale = this->networkConfig->_weightDecay * weight_update_param.decay_mult;
	//const Dtype learnScale = this->networkConfig->_baseLearningRate * weight_update_param.lr_mult;
	const Dtype learnScale = this->networkConfig->getLearningRate() * weight_update_param.lr_mult;
	_updateParam(weightSize, regScale, learnScale, _paramsHistory[Weight], _params[Weight]);

	const uint32_t biasSize = this->out_dim.rows;
	const Dtype regScale_b = this->networkConfig->_weightDecay * bias_update_param.decay_mult;
	//const Dtype learnScale_b = this->networkConfig->_baseLearningRate * bias_update_param.lr_mult;
	const Dtype learnScale_b = this->networkConfig->getLearningRate() * bias_update_param.lr_mult;
	_updateParam(biasSize, regScale_b, learnScale_b, _paramsHistory[Bias], _params[Bias]);
}



template <typename Dtype>
void FullyConnectedLayer<Dtype>::_updateParam(const uint32_t paramSize, const Dtype regScale,
    const Dtype learnScale, Data<Dtype>* dataHistory, Data<Dtype>* data) {
	const Dtype normScale = 1.0/this->in_dim.batches;
	const Dtype momentum = this->networkConfig->_momentum;
	const Dtype negativeOne = -1.0;

	//Data<Dtype>::printConfig = 1;
	data->print_grad("paramGrad:");
	dataHistory->print_data("paramHistoryData:");
	data->print_data("paramData:");

    data->mutable_host_grad();
	Dtype* d_paramGrad = data->mutable_device_grad();
	Dtype* d_paramData = data->mutable_device_data();
	Dtype* d_paramHistoryData = dataHistory->mutable_device_data();

	checkCudaErrors(cublasSscal(Cuda::cublasHandle, static_cast<int>(paramSize), &normScale,
        d_paramGrad, 1));								// normalize by batch size
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(paramSize), &regScale,
        d_paramData, 1, d_paramGrad, 1));				// regularize
	checkCudaErrors(cublasSscal(Cuda::cublasHandle, static_cast<int>(paramSize), &momentum,
        d_paramHistoryData, 1));						//
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(paramSize), &learnScale,
        d_paramGrad, 1, d_paramHistoryData, 1));		// momentum
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(paramSize), &negativeOne,
        d_paramHistoryData, 1, d_paramData, 1));		// update

	data->print_grad("paramGrad:");
	dataHistory->print_data("paramHistoryData:");
	data->print_data("paramData:");
	//Data<Dtype>::printConfig = 0;
}

template <typename Dtype>
void FullyConnectedLayer<Dtype>::applyChanges(LearnableLayer<Dtype> *targetLayer) {
    const uint32_t weightSize = this->in_dim.rows * this->out_dim.rows;
    const uint32_t biasSize = this->out_dim.rows;
    FullyConnectedLayer<Dtype>* _targetLayer = (FullyConnectedLayer<Dtype>*)targetLayer;

    int blockSize = BW;
    int gridSize;

    gridSize = (weightSize + blockSize -1) / blockSize;

    AddArrayOfFCLayer<<<gridSize, blockSize>>>(
        _targetLayer->_params[Weight]->mutable_device_grad(),
        _params[Weight]->device_grad(), weightSize);

    gridSize = (biasSize + blockSize -1) / blockSize;

    AddArrayOfFCLayer<<<gridSize, blockSize>>>(
        _targetLayer->_params[Bias]->mutable_device_grad(),
        _params[Bias]->device_grad(), biasSize);
}

template <typename Dtype>
void FullyConnectedLayer<Dtype>::syncParams(LearnableLayer<Dtype> *targetLayer) {
    const uint32_t weightSize = this->in_dim.rows * this->out_dim.rows;
    const uint32_t biasSize = this->out_dim.rows;
    FullyConnectedLayer<Dtype>* _targetLayer = (FullyConnectedLayer<Dtype>*)targetLayer;

    memcpy(_params[Weight]->mutable_host_grad(), _targetLayer->_params[Weight]->host_grad(),
        weightSize);
    memcpy(_params[Bias]->mutable_host_grad(), _targetLayer->_params[Bias]->host_grad(),
        biasSize);
#if 0
    for (uint32_t paramIdx = 0; paramIdx < weightSize; paramIdx++) {
        _params[Weight]->mutable_host_grad()[paramIdx] = 
            _targetLayer->_params[Weight]->host_grad()[paramIdx];
    }
    for (uint32_t paramIdx = 0; paramIdx < biasSize; paramIdx++) {
        _params[Bias]->mutable_host_grad()[paramIdx] = 
            _targetLayer->_params[Bias]->host_grad()[paramIdx];
    }
#endif
}


template <typename Dtype>
void FullyConnectedLayer<Dtype>::syncMutableMem() {
	_params[Weight]->mutable_device_grad();
	_params[Weight]->host_grad();
	_params[Bias]->mutable_device_grad();
	_params[Bias]->host_grad();
}

template <typename Dtype>
void FullyConnectedLayer<Dtype>::_feedforward() {
	/*
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
	*/

	_computeWeightedData();
	_computeWeightBiasedData();
	_computeActivatedData();

	//_dropoutForward();

}


template <typename Dtype>
void FullyConnectedLayer<Dtype>::_computeWeightedData() {
	_params[Weight]->print_data("weightData:");
	this->_inputData[0]->print_data("inputData:");

	// Apply weight to input data
	const Dtype* d_weightData = _params[Weight]->device_data();
	const Dtype* d_inputData = this->_inputData[0]->device_data();
	Dtype* d_preActivationData = _preActivation->mutable_device_data();

	checkCudaErrors(cublasSgemm(Cuda::cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
			this->out_dim.rows, this->out_dim.batches, this->in_dim.rows,
			&Cuda::alpha, d_weightData, this->out_dim.rows, d_inputData, this->in_dim.rows,
			&Cuda::beta, d_preActivationData, this->out_dim.rows));
}

template <typename Dtype>
void FullyConnectedLayer<Dtype>::_computeWeightBiasedData() {
	_preActivation->print_data("preActivationData:");
	_params[Bias]->print_data("biasData:");

	// Add bias to weighted input data
	const Dtype* d_biasData = _params[Bias]->device_data();
	Dtype* d_preActivationData = _preActivation->mutable_device_data();

	checkCudaErrors(cublasSgemm(Cuda::cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
			this->out_dim.rows, this->out_dim.batches, 1,
		&Cuda::alpha,
		d_biasData, this->out_dim.rows,
		d_onevec, 1,
		&Cuda::alpha,
		d_preActivationData, this->out_dim.rows));
}

template <typename Dtype>
void FullyConnectedLayer<Dtype>::_computeActivatedData() {
    // Activate weighted sum (+ bias)
    if (activation_fn) {
        const Dtype* d_preActivationData = _preActivation->device_data();
        Dtype* d_outputData = this->_outputData[0]->mutable_device_data();
        activation_fn->forward(this->outputTensorDesc, d_preActivationData, d_outputData);
    } else {
        this->_outputData[0]->set_device_data(_preActivation);
    }

    //Data<Dtype>::printConfig = true;
	_preActivation->print_data(this->name+string("/d_preActivationData:"));
	this->_outputData[0]->print_data(this->name+string("/d_outputData:"));
    //Data<Dtype>::printConfig = false;
}

template <typename Dtype>
void FullyConnectedLayer<Dtype>::_dropoutForward() {
	// TODO skip when test
	if(this->networkConfig->_status == NetworkStatus::Train && p_dropout < 1.0f) {
		int b_out = this->out_dim.batchsize();
		Dtype* h_mask_mem = _mask.mutable_host_mem();

		for(int i = 0; i < b_out; i++) {
			h_mask_mem[i] = ((rand()/(RAND_MAX+1.0) > p_dropout)?1:0);
		}
		//checkCudaErrors(cudaMemcpyAsync(d_mask, mask, sizeof(Dtype)*b_out, cudaMemcpyHostToDevice));
		//FillOnes<<<RoundUp(in_dim.batches, BW), BW>>>(this->d_onevec, in_dim.batches);


		const Dtype* d_mask_mem = _mask.device_mem();
		Dtype* d_outputData = this->_outputData[0]->mutable_device_data();

		Dropout<<<RoundUp(b_out, BW), BW>>>(b_out, d_outputData, d_mask_mem, 0, scale,
            d_outputData);

		//_mask.print("mask:");
		this->_outputData[0]->print_data("outputData:");
	}
}








template <typename Dtype>
void FullyConnectedLayer<Dtype>::_backpropagation() {
	//_dropoutBackward();

	_computePreActivationGrad();
	_computeWeightGrad();
	_computeBiasGrad();
	_computeInputGrad();

}



template <typename Dtype>
void FullyConnectedLayer<Dtype>::_dropoutBackward() {
	if(this->networkConfig->_status == NetworkStatus::Train && p_dropout < 1.0f) {
		this->_outputData[0]->print_grad("outputGrad:");
		const Dtype* d_mask_mem = _mask.device_mem();
		Dtype* d_outputGrad = this->_outputData[0]->mutable_device_grad();
		Dropout<<<RoundUp(this->out_dim.batchsize(), BW), BW>>>(this->out_dim.batchsize(),
            d_outputGrad, d_mask_mem, 0, scale, d_outputGrad);

		//_mask.print("mask:");
		this->_outputData[0]->print_grad("outputGrad:");
	}
}

template <typename Dtype>
void FullyConnectedLayer<Dtype>::_computePreActivationGrad() {
    if (activation_fn) {
        const Dtype* d_y = this->_outputData[0]->device_data();
        const Dtype* d_dy = this->_outputData[0]->device_grad();
        const Dtype* d_x = this->_preActivation->device_data();
        Dtype* d_dx = this->_preActivation->mutable_device_grad();
        this->activation_fn->backward(this->outputTensorDesc, d_y, d_dy, d_x, d_dx);
    }
    else {
        this->_preActivation->set_device_grad(this->_outputData[0]);
    }


    //if(this->name == "softmaxLayer") {
        //double sumsq = this->_preActivation->sumsq_device_grad();
        //cout << "preActivation grad sumsq: " << sumsq << endl;
    //  Data<Dtype>::printConfig = 1;
    //  this->_preActivation->print_grad("preActivationGrad:");
    //  Data<Dtype>::printConfig = 0;
    //}
}

template <typename Dtype>
void FullyConnectedLayer<Dtype>::_computeWeightGrad() {
	// d(Cost)/d(Weight)
	const Dtype* d_preActivationGrad = this->_preActivation->device_grad();
	const Dtype* d_inputData = this->_inputData[0]->device_data();
	Dtype* d_weightGrad = this->_params[Weight]->mutable_device_grad();

	checkCudaErrors(cublasSgemm(Cuda::cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
			this->out_dim.rows, this->in_dim.rows, this->out_dim.batches,
			&Cuda::alpha, d_preActivationGrad, this->out_dim.rows, d_inputData,
            this->in_dim.rows, &Cuda::beta, d_weightGrad, this->out_dim.rows));

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
	Dtype* d_inputGrad = this->_inputData[0]->mutable_device_grad();

	checkCudaErrors(cublasSgemm(Cuda::cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
			this->in_dim.rows, this->out_dim.batches, this->out_dim.rows,
			&Cuda::alpha, d_weightData, this->out_dim.rows, d_preActivationGrad,
            this->out_dim.rows, &Cuda::beta, d_inputGrad, this->in_dim.rows));
	this->_inputData[0]->print_grad("inputGrad:");

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


/*
template <typename Dtype>
double FullyConnectedLayer<Dtype>::testParamAbnormality() {
	const Dtype* weightGrad = _params[Weight]->host_grad();
	const size_t count = _params[Weight]->getCount();

	double mean = 0.0;
	for(uint32_t i = 0; i < count; i++) {
		mean += weightGrad[i];
	}
	mean /= count;

	double sd = 0.0;
	for(uint32_t i = 0; i < count; i++) {
		sd += (weightGrad[i]-mean)*(weightGrad[i]-mean);
	}
	sd = sqrt(sd)/(count-1);

	cout << this->name << ": mean: " << mean << ", sd: " << sd << endl;

	for(uint32_t i = 0; i < count; i++) {
		if(abs(weightGrad[i]-mean) > 10000*sd) {
			return weightGrad[i];
		}
	}
	return DBL_MAX;
}
*/







template FullyConnectedLayer<float>::~FullyConnectedLayer();
template void FullyConnectedLayer<float>::_shape(bool recursive);
template void FullyConnectedLayer<float>::_clearShape();
template void FullyConnectedLayer<float>::update();
template void FullyConnectedLayer<float>::_feedforward();
template void FullyConnectedLayer<float>::_backpropagation();


#endif








