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
	checkCUDNN(cudnnDestroyTensorDescriptor(inputTensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(outputTensorDesc));
	checkCudaErrors(cudaFree(d_onevec));

	ActivationFactory<Dtype>::destory(activation_fn);
}


template <typename Dtype>
void FullyConnectedLayer<Dtype>::shape() {
	Layer<Dtype>::_adjustInputShape();

	// 배치수가 변경되는 경우는 허용하도록 하자.
	const uint32_t count = Util::vecCountByAxis(this->_inputShape[0], 1);
	const uint32_t inputDataCount = this->_inputData[0]->getCountByAxis(1);
	if (inputDataCount == count)
		return;

	const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
	uint32_t batches = inputShape[0];
	uint32_t channels = 1;
	uint32_t in_rows = inputDataCount;
	uint32_t out_rows = this->n_out;
	uint32_t cols = 1;

	this->_inputShape[0] = {batches, channels, in_rows, cols};
	this->_preActivation->shape({batches, channels, out_rows, cols});
	this->_outputData[0]->shape({batches, channels, out_rows, cols});

	checkCUDNN(cudnnSetTensor4dDescriptor(
			this->inputTensorDesc,
			CUDNN_TENSOR_NCHW,
			CUDNN_DATA_FLOAT,
			batches, channels, in_rows, cols));

	checkCUDNN(cudnnSetTensor4dDescriptor(
			this->outputTensorDesc,
			CUDNN_TENSOR_NCHW,
			CUDNN_DATA_FLOAT,
			batches, channels, out_rows, cols));

	printf("<%s> layer' input-0 has reshaped as: %dx%dx%dx%d\n",
			this->name.c_str(), batches, channels, in_rows, cols);
	printf("<%s> layer' output-0 has reshaped as: %dx%dx%dx%d\n",
			this->name.c_str(), batches, channels, out_rows, cols);

	uint32_t u_in = in_rows;
	uint32_t u_out = out_rows;
	uint32_t b_in = batches * in_rows;
	uint32_t b_out = batches * out_rows;

	_params[ParamType::Weight]->shape({1, 1, u_out, u_in});
	_params[ParamType::Bias]->shape({1, u_out, 1, 1});
	_paramsHistory[ParamType::Weight]->shape({1, 1, u_out, u_in});
	_paramsHistory[ParamType::Bias]->shape({1, u_out, 1, 1});

	weight_filler.fill(_params[ParamType::Weight]);
	bias_filler.fill(_params[ParamType::Bias]);

	_params[ParamType::Weight]->print_data("weight:");
	_params[ParamType::Bias]->print_data("bias:");

	checkCudaErrors(Util::ucudaMalloc(&this->d_onevec, sizeof(Dtype)*batches));
	FillValues<<<RoundUp(batches, BW), BW>>>(this->d_onevec, batches, 1.0f);

	_mask.shape(b_out);


	/*
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
	*/
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
	const uint32_t in_rows = this->_inputShape[0][2];
	const uint32_t out_rows = this->_outputData[0]->getShape(2);

	const uint32_t weightSize = in_rows * out_rows;
	const Dtype regScale = this->networkConfig->_weightDecay * weight_update_param.decay_mult;
	//const Dtype learnScale = this->networkConfig->_baseLearningRate * weight_update_param.lr_mult;
	const Dtype learnScale = this->networkConfig->getLearningRate() * weight_update_param.lr_mult;
	_updateParam(weightSize, regScale, learnScale, _paramsHistory[Weight], _params[Weight]);

	const uint32_t biasSize = out_rows;
	const Dtype regScale_b = this->networkConfig->_weightDecay * bias_update_param.decay_mult;
	//const Dtype learnScale_b = this->networkConfig->_baseLearningRate * bias_update_param.lr_mult;
	const Dtype learnScale_b = this->networkConfig->getLearningRate() * bias_update_param.lr_mult;
	_updateParam(biasSize, regScale_b, learnScale_b, _paramsHistory[Bias], _params[Bias]);
}



template <typename Dtype>
void FullyConnectedLayer<Dtype>::_updateParam(const uint32_t paramSize, const Dtype regScale,
		const Dtype learnScale, Data<Dtype>* dataHistory, Data<Dtype>* data) {
	const uint32_t batches = this->_inputShape[0][0];
	const Dtype normScale = 1.0/batches;
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

	checkCudaErrors(cublasSscal(Cuda::cublasHandle, static_cast<int>(paramSize), &normScale, d_paramGrad, 1));								// normalize by batch size
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(paramSize), &regScale, d_paramData, 1, d_paramGrad, 1));				// regularize
	checkCudaErrors(cublasSscal(Cuda::cublasHandle, static_cast<int>(paramSize), &momentum, d_paramHistoryData, 1));						//
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(paramSize), &learnScale, d_paramGrad, 1, d_paramHistoryData, 1));		// momentum
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(paramSize), &negativeOne, d_paramHistoryData, 1, d_paramData, 1));		// update

	data->print_grad("paramGrad:");
	dataHistory->print_data("paramHistoryData:");
	data->print_data("paramData:");
	//Data<Dtype>::printConfig = 0;
}

template <typename Dtype>
void FullyConnectedLayer<Dtype>::applyChanges(LearnableLayer<Dtype> *targetLayer) {
	const uint32_t in_rows = this->_inputShape[0][2];
	const uint32_t out_rows = this->_outputData[0]->getShape(2);

    const uint32_t weightSize = in_rows * out_rows;
    const uint32_t biasSize = out_rows;
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
	const uint32_t in_rows = this->_inputShape[0][2];
	const uint32_t out_rows = this->_outputData[0]->getShape(2);

    const uint32_t weightSize = in_rows * out_rows;
    const uint32_t biasSize = out_rows;
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

	//Data<Dtype>::printConfig = true;
	_computeWeightedData();
	_computeWeightBiasedData();
	_computeActivatedData();
	//Data<Dtype>::printConfig = false;

	//_dropoutForward();

}


template <typename Dtype>
void FullyConnectedLayer<Dtype>::_computeWeightedData() {
	const uint32_t batches = this->_inputShape[0][0];
	const uint32_t in_rows = this->_inputShape[0][2];
	const uint32_t out_rows = this->_outputData[0]->getShape(2);

	// Apply weight to input data
	const Dtype* d_weightData = _params[Weight]->device_data();
	const Dtype* d_inputData = this->_inputData[0]->device_data();
	Dtype* d_preActivationData = _preActivation->mutable_device_data();

	_params[Weight]->print_data();
	this->_inputData[0]->print_data();
	this->_inputData[0]->print_data_flatten();

	checkCudaErrors(cublasSgemm(Cuda::cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
			out_rows, batches, in_rows,
			&Cuda::alpha, d_weightData, out_rows, d_inputData, in_rows,
			&Cuda::beta, d_preActivationData, out_rows));

	_preActivation->print_data();
}

template <typename Dtype>
void FullyConnectedLayer<Dtype>::_computeWeightBiasedData() {
	const uint32_t batches = this->_inputShape[0][0];
	const uint32_t in_rows = this->_inputShape[0][2];
	const uint32_t out_rows = this->_outputData[0]->getShape(2);

	// Add bias to weighted input data
	const Dtype* d_biasData = _params[Bias]->device_data();
	Dtype* d_preActivationData = _preActivation->mutable_device_data();

	_params[Bias]->print_data();

	checkCudaErrors(cublasSgemm(Cuda::cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
			out_rows, batches, 1,
			&Cuda::alpha,
			d_biasData, out_rows,
			d_onevec, 1,
			&Cuda::alpha,
			d_preActivationData, out_rows));

	_params[Bias]->print_data();
}

template <typename Dtype>
void FullyConnectedLayer<Dtype>::_computeActivatedData() {
	// Activate weighted sum (+ bias)
	const Dtype* d_preActivationData = _preActivation->mutable_device_data();
	Dtype* d_outputData = this->_outputData[0]->mutable_device_data();

	_preActivation->print_data();

	activation_fn->forward(this->outputTensorDesc, d_preActivationData, d_outputData);

	this->_outputData[0]->print_data();
}

template <typename Dtype>
void FullyConnectedLayer<Dtype>::_dropoutForward() {
	// TODO skip when test
	if(this->networkConfig->_status == NetworkStatus::Train && p_dropout < 1.0f) {
		//int b_out = this->out_dim.batchsize();
		int b_out = this->_outputData[0]->getCount();
		Dtype* h_mask_mem = _mask.mutable_host_mem();

		for(int i = 0; i < b_out; i++) {
			h_mask_mem[i] = ((rand()/(RAND_MAX+1.0) > p_dropout)?1:0);
		}

		const Dtype* d_mask_mem = _mask.device_mem();
		Dtype* d_outputData = this->_outputData[0]->mutable_device_data();

		Dropout<<<RoundUp(b_out, BW), BW>>>(b_out, d_outputData, d_mask_mem, 0, scale, d_outputData);
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
		const uint32_t batchSize = this->_inputData[0]->getCount();

		this->_outputData[0]->print_grad("outputGrad:");
		const Dtype* d_mask_mem = _mask.device_mem();
		Dtype* d_outputGrad = this->_outputData[0]->mutable_device_grad();
		Dropout<<<RoundUp(batchSize, BW), BW>>>(batchSize, d_outputGrad, d_mask_mem, 0, scale, d_outputGrad);

		//_mask.print("mask:");
		this->_outputData[0]->print_grad("outputGrad:");
	}
}







template <typename Dtype>
void FullyConnectedLayer<Dtype>::_computePreActivationGrad() {
	this->_outputData[0]->print_data("output:");

	const Dtype* d_y = this->_outputData[0]->device_data();
	const Dtype* d_dy = this->_outputData[0]->device_grad();
	const Dtype* d_x = this->_preActivation->device_data();
	Dtype* d_dx = this->_preActivation->mutable_device_grad();

	this->activation_fn->backward(this->outputTensorDesc, d_y, d_dy, d_x, d_dx);


	//if(this->name == "softmaxLayer") {
		//double sumsq = this->_preActivation->sumsq_device_grad();
		//cout << "preActivation grad sumsq: " << sumsq << endl;
	//	Data<Dtype>::printConfig = 1;
	//	this->_preActivation->print_grad("preActivationGrad:");
	//	Data<Dtype>::printConfig = 0;
	//}
}



template <typename Dtype>
void FullyConnectedLayer<Dtype>::_computeWeightGrad() {
	const uint32_t batches = this->_inputShape[0][0];
	const uint32_t in_rows = this->_inputShape[0][2];
	const uint32_t out_rows = this->_outputData[0]->getShape(2);

	// d(Cost)/d(Weight)
	const Dtype* d_preActivationGrad = this->_preActivation->device_grad();
	const Dtype* d_inputData = this->_inputData[0]->device_data();
	Dtype* d_weightGrad = this->_params[Weight]->mutable_device_grad();

	checkCudaErrors(cublasSgemm(Cuda::cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
			out_rows, in_rows, batches,
			&Cuda::alpha, d_preActivationGrad, out_rows, d_inputData, in_rows,
			&Cuda::beta, d_weightGrad, out_rows));

}

template <typename Dtype>
void FullyConnectedLayer<Dtype>::_computeBiasGrad() {
	const uint32_t batches = this->_inputShape[0][0];
	const uint32_t in_rows = this->_inputShape[0][2];
	const uint32_t out_rows = this->_outputData[0]->getShape(2);

	// d(Cost)/d(Bias) (same as d_preActivationGrad)
	const Dtype* d_preActivationGrad = this->_preActivation->device_grad();
	Dtype* d_biasGrad = _params[Bias]->mutable_device_grad();

	checkCudaErrors(cublasSgemv(Cuda::cublasHandle, CUBLAS_OP_N,
			out_rows, batches,
			&Cuda::alpha, d_preActivationGrad, out_rows, d_onevec, 1,
			&Cuda::beta, d_biasGrad, 1));
	_params[Bias]->print_grad("biasGrad:");
	_params[Weight]->print_data("weightData:");
	_preActivation->print_grad("preActivationGrad");
}

template <typename Dtype>
void FullyConnectedLayer<Dtype>::_computeInputGrad() {
	const uint32_t batches = this->_inputShape[0][0];
	const uint32_t in_rows = this->_inputShape[0][2];
	const uint32_t out_rows = this->_outputData[0]->getShape(2);

	// d(Cost)/d(Input)
	const Dtype* d_weightData = _params[Weight]->device_data();
	const Dtype* d_preActivationGrad = this->_preActivation->device_grad();
	Dtype* d_inputGrad = this->_inputData[0]->mutable_device_grad();

	checkCudaErrors(cublasSgemm(Cuda::cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
			in_rows, batches, out_rows,
			&Cuda::alpha, d_weightData, out_rows, d_preActivationGrad, out_rows,
			&Cuda::beta, d_inputGrad, in_rows));
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
template void FullyConnectedLayer<float>::shape();
template void FullyConnectedLayer<float>::_clearShape();
template void FullyConnectedLayer<float>::update();
template void FullyConnectedLayer<float>::_feedforward();
template void FullyConnectedLayer<float>::_backpropagation();


#endif








