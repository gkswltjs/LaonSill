/*
 * ConvLayer.cpp
 *
 *  Created on: 2016. 5. 23.
 *      Author: jhkim
 */


#ifdef GPU_MODE
#include "ConvLayer.h"
#include "FullyConnectedLayer.h"
#include "Util.h"
#include "Exception.h"
#include "NetworkConfig.h"
#include "cuda_runtime.h"
#include "MathFunctions.h"
#include <algorithm>

#define CONVLAYER_LOG 0

using namespace std;

/**
 * dst array에 src array를 더한다.
 *
 * @param dst dst array, dst + src가 저장이 될 장소
 * @param src src array
 * @param N The number of elements in the array.
 */
template <typename Dtype>
__global__ void AddArrayOfConvLayer(Dtype* dst, const Dtype* src, int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= N)
		return;

	dst[idx] = dst[idx] + src[idx];
}

template <typename Dtype>
ConvLayer<Dtype>::~ConvLayer() {
	delete this->_params[ParamType::Filter];
	delete this->_params[ParamType::Bias];
	this->_params.clear();

	delete this->_paramsHistory[ParamType::Filter];
	delete this->_paramsHistory[ParamType::Bias];
	this->_paramsHistory.clear();

	//delete _preActivation;

	if(d_workspace) checkCudaErrors(cudaFree(d_workspace));

	checkCUDNN(cudnnDestroyTensorDescriptor(inputTensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(outputTensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(biasTensorDesc));
	checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc));
	checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));

	//ActivationFactory<Dtype>::destory(activation_fn);
}

template <typename Dtype>
void ConvLayer<Dtype>::initialize(filter_dim filter_d, update_param weight_update_param,
    update_param bias_update_param, param_filler<Dtype> weight_filler, 
    param_filler<Dtype> bias_filler) {

	this->type = Layer<Dtype>::Conv;
	this->filter_d = filter_d;

	this->weight_update_param = weight_update_param;
	this->bias_update_param = bias_update_param;
	this->weight_filler = weight_filler;
	this->bias_filler = bias_filler;

	const int filter_size = filter_d.size();

	this->_params.resize(2);
	this->_params[Filter] = new Data<Dtype>(this->name + "_filter");
	this->_params[Bias] = new Data<Dtype>(this->name + "_bias");
	this->_params[Filter]->reshape(
        {filter_d.filters, filter_d.channels, filter_d.rows, filter_d.cols});
	this->_params[Bias]->reshape({filter_d.filters, 1, 1, 1});

    weight_filler.fill(this->_params[Filter]);
    bias_filler.fill(this->_params[Bias]);

	this->_paramsHistory.resize(2);
	this->_paramsHistory[Filter] = new Data<Dtype>(this->name + "_filter_history");
	this->_paramsHistory[Bias] = new Data<Dtype>(this->name + "_bias_history");
	this->_paramsHistory[Filter]->reshape(
        {filter_d.filters, filter_d.channels, filter_d.rows, filter_d.cols});
	this->_paramsHistory[Bias]->reshape({filter_d.filters, 1, 1, 1});

	this->_paramsInitialized.resize(2);
	this->_paramsInitialized[Filter] = false;
	this->_paramsInitialized[Bias] = false;

	checkCUDNN(cudnnCreateTensorDescriptor(&inputTensorDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&outputTensorDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&biasTensorDesc));
	checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
	checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

	checkCUDNN(cudnnSetTensor4dDescriptor(biasTensorDesc,
			CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
			1, filter_d.filters, 1, 1));

	checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc,
			CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
			filter_d.filters, filter_d.channels, filter_d.rows, filter_d.cols));

	//int pad = (filter_d.rows-1)/2;
	checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc,
			filter_d.pad, filter_d.pad, filter_d.stride, filter_d.stride, 1, 1,
			CUDNN_CROSS_CORRELATION));

	//this->activation_fn = ActivationFactory<Dtype>::create(activationType);

	this->d_workspace = 0;
}

template <typename Dtype>
void ConvLayer<Dtype>::reshape() {
	Layer<Dtype>::_adjustInputShape();

	if (!Layer<Dtype>::_isInputShapeChanged(0))
		return;

	const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
	uint32_t batches 	= inputShape[0];
	uint32_t channels 	= inputShape[1];
	uint32_t rows 		= inputShape[2];
	uint32_t cols 		= inputShape[3];

	checkCUDNN(cudnnSetTensor4dDescriptor(
			this->inputTensorDesc,
			CUDNN_TENSOR_NCHW,
			CUDNN_DATA_FLOAT,
			batches, channels, rows, cols));

	int n = 0, c = 0, h = 0, w = 0;
	checkCUDNN(cudnnGetConvolution2dForwardOutputDim(
			this->convDesc,
			this->inputTensorDesc,
			this->filterDesc,
			&n, &c, &h, &w));

	checkCUDNN(cudnnSetTensor4dDescriptor(
			this->outputTensorDesc,
			CUDNN_TENSOR_NCHW,
			CUDNN_DATA_FLOAT,
			n, c, h, w));

	const uint32_t obatches = static_cast<uint32_t>(n);
	const uint32_t ochannels = static_cast<uint32_t>(c);
	const uint32_t orows = static_cast<uint32_t>(h);
	const uint32_t ocols = static_cast<uint32_t>(w);

#if !CONVLAYER_LOG
	printf("<%s> layer' output-0 has reshaped as: %dx%dx%dx%d\n",
			this->name.c_str(), obatches, ochannels, orows, ocols);
#endif

	this->_inputShape[0] = inputShape;
	//this->_preActivation->reshape({obatches, ochannels, orows, ocols});
	this->_outputData[0]->reshape({obatches, ochannels, orows, ocols});


	//int u_in = this->in_dim.unitsize();
	//int u_out = this->out_dim.unitsize();
	//int b_in = this->in_dim.batchsize();
	//int b_out = this->out_dim.batchsize();
	const int u_in = channels * rows * cols;
	const int u_out = c * h * w;
	const int b_in = batches * channels * rows * cols;
	const int b_out = n * c * h * w;

	size_t convFwdWorkspaceSize;
	size_t convBwdFilterWorkspaceSize;
	size_t convBwdDataWorkspaceSize;

	// forward algorithm
	checkCUDNN(cudnnGetConvolutionForwardAlgorithm(
			Cuda::cudnnHandle,
			this->inputTensorDesc,
			this->filterDesc,
			this->convDesc,
			this->outputTensorDesc,
			//CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
			CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
			8<<20,
			&convFwdAlgo));

	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
			Cuda::cudnnHandle,
			this->inputTensorDesc,
			this->filterDesc,
			this->convDesc,
			this->outputTensorDesc,
			this->convFwdAlgo,
			&convFwdWorkspaceSize));

	// backward filter algorithm
	checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(
			Cuda::cudnnHandle,
			this->inputTensorDesc,
			this->outputTensorDesc,
			this->convDesc,
			this->filterDesc,
			//CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
			CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
			8<<20,
			&this->convBwdFilterAlgo));

	checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
			Cuda::cudnnHandle,
			this->inputTensorDesc,
			this->outputTensorDesc,
			this->convDesc,
			this->filterDesc,
			this->convBwdFilterAlgo,
			&convBwdFilterWorkspaceSize));

	// backward data algorithm
	checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(
			Cuda::cudnnHandle,
			this->filterDesc,
			this->outputTensorDesc,
			this->convDesc,
			this->inputTensorDesc,
			//CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
			CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
			8<<20,
			&convBwdDataAlgo));

	checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
			Cuda::cudnnHandle,
			this->filterDesc,
			this->outputTensorDesc,
			this->convDesc,
			this->inputTensorDesc,
			this->convBwdDataAlgo,
			&convBwdDataWorkspaceSize));

	workspaceSize = 0;
	workspaceSize = max(workspaceSize, convFwdWorkspaceSize);
	workspaceSize = max(workspaceSize, convBwdFilterWorkspaceSize);
	workspaceSize = max(workspaceSize, convBwdDataWorkspaceSize);


	if(workspaceSize > 0) {
		cout << this->name << "'s workspace: " << workspaceSize << endl;
		if (this->d_workspace) {
			checkCudaErrors(cudaFree(d_workspace));
			this->d_workspace = 0;
		}
		checkCudaErrors(Util::ucudaMalloc(&d_workspace, workspaceSize));
	}
}

template <typename Dtype>
void ConvLayer<Dtype>::update() {
	// update filters ...
	const uint32_t weightSize = filter_d.size();
	const Dtype regScale = this->networkConfig->_weightDecay * weight_update_param.decay_mult;
	const Dtype learnScale = 
        this->networkConfig->getLearningRate() * weight_update_param.lr_mult;
	_updateParam(weightSize, regScale, learnScale, this->_paramsHistory[Filter], this->_params[Filter]);

	// update biases ...
	const uint32_t biasSize = filter_d.filters;
	const Dtype regScale_b = this->networkConfig->_weightDecay * bias_update_param.decay_mult;
	const Dtype learnScale_b = 
        this->networkConfig->getLearningRate() * bias_update_param.lr_mult;
	_updateParam(biasSize, regScale_b, learnScale_b, this->_paramsHistory[Bias], this->_params[Bias]);
}

template <typename Dtype>
void ConvLayer<Dtype>::_updateParam(const uint32_t paramSize, const Dtype regScale,
    const Dtype learnScale, Data<Dtype>* dataHistory, Data<Dtype>* data) {
	//const uint32_t batches = this->_inputData[0]->getShape(0);
	//const Dtype normScale = 1.0/batches;
	const Dtype momentum = this->networkConfig->_momentum;
	const Dtype negativeOne = -1.0;

    data->mutable_host_grad();
	Dtype* d_paramGrad = data->mutable_device_grad();   // should update grad
	Dtype* d_paramData = data->mutable_device_data();
	Dtype* d_paramHistoryData = dataHistory->mutable_device_data();


	/*
	//checkCudaErrors(cublasSscal(Cuda::cublasHandle, static_cast<int>(paramSize),
    //    &normScale, d_paramGrad, 1));	// normalized by batch size
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(paramSize), &regScale,
        d_paramData, 1, d_paramGrad, 1));	// regularize
	checkCudaErrors(cublasSscal(Cuda::cublasHandle, static_cast<int>(paramSize), &momentum,
        d_paramHistoryData, 1));
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(paramSize), &learnScale,
        d_paramGrad, 1, d_paramHistoryData, 1));	// momentum
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(paramSize), &negativeOne,
        d_paramHistoryData, 1, d_paramData, 1));	// update
        */
	// grad = grad + reg * data


	//Data<Dtype>::printConfig = true;
	//SyncMem<Dtype>::printConfig = true;

	data->print_data({}, false);
	data->print_grad({}, false);
	dataHistory->print_data({}, false);



	soooa_gpu_axpy(static_cast<int>(paramSize), regScale, d_paramData, d_paramGrad);
	data->print_grad({}, false);



	soooa_gpu_axpby(static_cast<int>(paramSize), learnScale, d_paramGrad, momentum,
			d_paramHistoryData);
	dataHistory->print_data({}, false);



	soooa_copy(static_cast<int>(paramSize), d_paramHistoryData, d_paramGrad);
	data->print_grad({}, false);

	// update
	soooa_gpu_axpy(static_cast<int>(paramSize), negativeOne, d_paramGrad, d_paramData);
	data->print_data({}, false);


	//Data<Dtype>::printConfig = false;
	//SyncMem<Dtype>::printConfig = false;

}

template <typename Dtype>
void ConvLayer<Dtype>::applyChanges(LearnableLayer<Dtype> *targetLayer) {
	const uint32_t weightSize = this->filter_d.size();
	const uint32_t biasSize = this->filter_d.filters;
    ConvLayer<Dtype>* _targetLayer = (ConvLayer<Dtype>*)targetLayer;

    //int blockSize = BW;
    int blockSize = SOOOA_CUDA_NUM_THREADS;
    int gridSize = (weightSize + blockSize -1) / blockSize;

    AddArrayOfConvLayer<<<gridSize, blockSize>>>(
        _targetLayer->_params[Filter]->mutable_device_grad(),
        this->_params[Filter]->device_grad(), weightSize);

    gridSize = (biasSize + blockSize -1) / blockSize;

    AddArrayOfConvLayer<<<gridSize, blockSize>>>(
        _targetLayer->_params[Bias]->mutable_device_grad(),
        this->_params[Bias]->device_grad(), biasSize);
}

template <typename Dtype>
void ConvLayer<Dtype>::syncParams(LearnableLayer<Dtype> *targetLayer) {
	const uint32_t weightSize = filter_d.size();
	const uint32_t biasSize = filter_d.filters;
    ConvLayer<Dtype>* _targetLayer = (ConvLayer<Dtype>*)targetLayer;

    memcpy(this->_params[Filter]->mutable_host_grad(), _targetLayer->_params[Filter]->host_grad(),
        weightSize);
    memcpy(this->_params[Bias]->mutable_host_grad(), _targetLayer->_params[Bias]->host_grad(),
        biasSize);
}

template <typename Dtype>
void ConvLayer<Dtype>::syncMutableMem() {
	this->_params[Filter]->mutable_device_grad();
	this->_params[Filter]->host_grad();
	this->_params[Bias]->mutable_device_grad();
	this->_params[Bias]->host_data();
}

template <typename Dtype>
void ConvLayer<Dtype>::feedforward() {
	reshape();

	_computeFiltersConvolutionData();
}

template <typename Dtype>
void ConvLayer<Dtype>::_computeFiltersConvolutionData() {
	// Apply filters to input data
	const Dtype* d_inputData = this->_inputData[0]->device_data();
	const Dtype* d_filtersData = this->_params[Filter]->device_data();
	//Dtype* d_preActivationData = _preActivation->mutable_device_data();
	Dtype* d_outputData = this->_outputData[0]->mutable_device_data();

#if CONVLAYER_LOG
	this->_inputData[0]->print_data();
	this->_params[Filter]->print_data();
#endif

	checkCUDNN(cudnnConvolutionForward(Cuda::cudnnHandle,
			&Cuda::alpha, this->inputTensorDesc, d_inputData, filterDesc, d_filtersData,
            convDesc, convFwdAlgo, d_workspace, workspaceSize,
			&Cuda::beta, this->outputTensorDesc, d_outputData));

	const Dtype* d_biasesData = this->_params[Bias]->device_data();
	checkCUDNN(cudnnAddTensor(Cuda::cudnnHandle,
			&Cuda::alpha, biasTensorDesc, d_biasesData,
			&Cuda::alpha, this->outputTensorDesc, d_outputData));
}

/*
template <typename Dtype>
void ConvLayer<Dtype>::_computeActivationData() {
	// Activate filtered result
	if (activation_fn) {
		const Dtype* d_preActivationData = _preActivation->device_data();
		Dtype* d_output = this->_outputData[0]->mutable_device_data();
		activation_fn->forward(this->outputTensorDesc, d_preActivationData, d_output);
	} else {
		this->_outputData[0]->set_device_data(_preActivation);
	}

#if CONVLAYER_LOG
	_preActivation->print_data();
	this->_outputData[0]->print_data();
#endif
}
*/



template <typename Dtype>
void ConvLayer<Dtype>::backpropagation() {
	if (this->_propDown[0]) {
		//_computePreActivationGrad();
		_computeFiltersGrad();
		_computeBiasesGrad();
		_computeInputGrad();
	}
}


/*
template <typename Dtype>
void ConvLayer<Dtype>::_computePreActivationGrad() {
#if CONVLAYER_LOG
	this->_outputData[0]->print_grad();
	this->_outputData[0]->print_data();
#endif

	if (activation_fn) {
		const Dtype* d_outputData = this->_outputData[0]->device_data();
		const Dtype* d_outputGrad = this->_outputData[0]->device_grad();
		const Dtype* d_preActivationData = _preActivation->device_data();
		Dtype* d_preActivationGrad = _preActivation->mutable_device_grad();

		activation_fn->backward(this->outputTensorDesc,
				d_outputData, d_outputGrad, d_preActivationData, d_preActivationGrad);
	} else {
		this->_preActivation->set_device_grad(this->_outputData[0]);
	}
}
*/



template <typename Dtype>
void ConvLayer<Dtype>::_computeFiltersGrad() {
#if CONVLAYER_LOG
	this->_inputData[0]->print_data("inputData:");
	this->_preActivation->print_grad("preActivationGrad:");
#endif

	// d(Cost)/d(Filters)
	const Dtype* d_inputData = this->_inputData[0]->device_data();
	//const Dtype* d_preActivationGrad = this->_preActivation->device_grad();
	const Dtype* d_outputGrad = this->_outputData[0]->device_grad();
	Dtype* d_filtersGrad = this->_params[Filter]->mutable_device_grad();

	checkCUDNN(cudnnConvolutionBackwardFilter(Cuda::cudnnHandle,
			&Cuda::alpha, this->inputTensorDesc, d_inputData, this->outputTensorDesc,
            d_outputGrad, convDesc, convBwdFilterAlgo, d_workspace, workspaceSize,
			&Cuda::beta, filterDesc, d_filtersGrad));

#if CONVLAYER_LOG
	this->_params[Filter]->print_grad("filtersGrad:");
#endif
}

template <typename Dtype>
void ConvLayer<Dtype>::_computeBiasesGrad() {
	// d(Cost)/d(Biases)
	//const Dtype* d_preActivationGrad = this->_preActivation->device_grad();
	const Dtype* d_outputGrad = this->_outputData[0]->device_grad();
	Dtype* d_biasGrad = this->_params[Bias]->mutable_device_grad();

	checkCUDNN(cudnnConvolutionBackwardBias(Cuda::cudnnHandle,
			&Cuda::alpha, this->outputTensorDesc, d_outputGrad,
			&Cuda::beta, biasTensorDesc, d_biasGrad));
}

template <typename Dtype>
void ConvLayer<Dtype>::_computeInputGrad() {
	// d(Cost)/d(Input)
	const Dtype* d_filtersData = this->_params[Filter]->device_data();
	//const Dtype* d_preActivationGrad = this->_preActivation->device_grad();
	const Dtype* d_outputGrad = this->_outputData[0]->device_grad();
	Dtype* d_inputGrad = this->_inputData[0]->mutable_device_grad();
	checkCUDNN(cudnnConvolutionBackwardData(Cuda::cudnnHandle,
			&Cuda::alpha, filterDesc, d_filtersData, this->outputTensorDesc,
            d_outputGrad, convDesc, convBwdDataAlgo, d_workspace, workspaceSize,
			&Cuda::beta, this->inputTensorDesc, d_inputGrad));

#if CONVLAYER_LOG
	this->_inputData[0]->print_grad("inputGrad:");
	this->_params[Filter]->print_data("filtersData:");
#endif
}

template ConvLayer<float>::~ConvLayer();
template void ConvLayer<float>::initialize(filter_dim filter_d,
    update_param weight_update_param, update_param bias_update_param,
    param_filler<float> weight_filler, param_filler<float> bias_filler);
template void ConvLayer<float>::reshape();
template void ConvLayer<float>::update();
template void ConvLayer<float>::feedforward();
template void ConvLayer<float>::backpropagation();

#endif
