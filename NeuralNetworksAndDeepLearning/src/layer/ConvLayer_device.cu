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
#include "PropMgmt.h"

#define CONVLAYER_LOG 1

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
__global__ void DoNesterov(int size, const Dtype* dx, Dtype* v_prev, Dtype* v, Dtype* x,
    const Dtype mu, const Dtype lr)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

    /****
     * Nesterov Alogorithm
     *
     * v_prev = v # back this up
     * v = mu * v - learning_rate * dx # velocity update stays the same
     * x += -mu * v_prev + (1 + mu) * v # position update changes form
     *
     */

    v_prev[idx] = v[idx];
    v[idx] = mu * v[idx] - lr * dx[idx];
    x[idx] += (-1.0) * mu * v_prev[idx] + (1 + mu) * v[idx];
}

template <typename Dtype>
__global__ void DoAdagrad(int size, const Dtype* dx, Dtype* cache, Dtype* x,
    const Dtype lr, const Dtype eps)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

    /****
     * Adagrad Alogorithm
     *
     * cache += dx**2
     * x += -learning_rate * dx / (sqrt(cache) + eps)
     *
     */

    cache[idx] += dx[idx] * dx[idx];
    x[idx] += (-1.0) * lr * dx[idx] / (sqrtf(cache[idx]) + eps);
}

template <typename Dtype>
__global__ void DoRMSprop(int size, const Dtype* dx, Dtype* cache, Dtype* x,
    const Dtype lr, const Dtype eps, const Dtype dr)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

    /****
     * RMSprop
     *
     * cache = decay_rate * cache + (1 - decay_rate) * dx**2
     * x += - learning_rate * dx / (sqrt(cache) + eps)
     *
     */

    cache[idx] = dr * cache[idx] + (1.0 - dr) * dx[idx] * dx[idx];
    x[idx] += (-1.0) * lr * dx[idx] / (sqrtf(cache[idx]) + eps);
}

#define USE_TENSORFLOW_ADAM         0

template <typename Dtype>
__global__ void DoAdam(int size, const Dtype* dx, Dtype* m, Dtype* v, Dtype* x,
    const Dtype lr, const Dtype eps, const Dtype beta1, const Dtype beta2,
    const Dtype decayedBeta1, const Dtype decayedBeta2)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

    /****
     * Adam
     *
     * m = beta1 * m + (1 - beta1) * dx
     * v = beta2 * v + (1 - beta2) * (dx**2)
     * x += -learning_rate * m / (sqrt(v) + eps)
     *
     */
    m[idx] = beta1 * m[idx] + (1.0 - beta1) * dx[idx];
    v[idx] = beta2 * v[idx] + (1.0 - beta2) * dx[idx] * dx[idx];
#if USE_TENSORFLOW_ADAM
    Dtype learningRate = lr * sqrtf(1.0 - decayedBeta2) / (1.0 - decayedBeta1);
    x[idx] += (-1.0) * learningRate * m[idx] / (sqrtf(v[idx]) + eps);
#else
    x[idx] += (-1.0) * lr * m[idx] / (sqrtf(v[idx]) + eps);
#endif
}



template <typename Dtype>
ConvLayer<Dtype>::~ConvLayer() {
    if (this->isReceiver) {
        Donator<Dtype>::releaseReceiver(this->donatorID);
    } else {
        delete this->_params[ParamType::Filter];
        delete this->_params[ParamType::Bias];
        this->_params.clear();

        delete this->_paramsHistory[ParamType::Filter];
        delete this->_paramsHistory[ParamType::Bias];
        delete this->_paramsHistory2[ParamType::Filter];
        delete this->_paramsHistory2[ParamType::Bias];
        
        this->_paramsHistory.clear();
        this->_paramsHistory2.clear();
    }

	if(this->d_workspace) checkCudaErrors(cudaFree(this->d_workspace));

	checkCUDNN(cudnnDestroyTensorDescriptor(this->inputTensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(this->outputTensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(this->biasTensorDesc));
	checkCUDNN(cudnnDestroyFilterDescriptor(this->filterDesc));
	checkCUDNN(cudnnDestroyConvolutionDescriptor(this->convDesc));
}

template <typename Dtype>
void ConvLayer<Dtype>::initialize() {
	const string& name = SLPROP_BASE(name);
	const filter_dim& filterDim = SLPROP(Conv, filterDim);
	param_filler<Dtype>& weightFiller = SLPROP(Conv, weightFiller);
	param_filler<Dtype>& biasFiller = SLPROP(Conv, biasFiller);
	const bool deconv = SLPROP(Conv, deconv);

    if (!deconv)
	    this->type = Layer<Dtype>::Conv;
    else
	    this->type = Layer<Dtype>::Deconv;
	const int filter_size = filterDim.size();

	this->_params.resize(2);
	this->_params[Filter] = new Data<Dtype>(name + "_filter");
	this->_params[Bias] = new Data<Dtype>(name + "_bias");
	this->_params[Filter]->reshape(
        {filterDim.filters, filterDim.channels, filterDim.rows, filterDim.cols});
	this->_params[Bias]->reshape({filterDim.filters, 1, 1, 1});

    weightFiller.fill(this->_params[Filter]);
    biasFiller.fill(this->_params[Bias]);

	this->_paramsHistory.resize(2);
	this->_paramsHistory[Filter] = new Data<Dtype>(name + "_filter_history");
	this->_paramsHistory[Bias] = new Data<Dtype>(name + "_bias_history");
	this->_paramsHistory[Filter]->reshape(
        {filterDim.filters, filterDim.channels, filterDim.rows, filterDim.cols});
	this->_paramsHistory[Bias]->reshape({filterDim.filters, 1, 1, 1});

	this->_paramsHistory2.resize(2);
	this->_paramsHistory2[Filter] = new Data<Dtype>(name + "_filter_history2");
	this->_paramsHistory2[Bias] = new Data<Dtype>(name + "_bias_history2");
	this->_paramsHistory2[Filter]->reshape(
        {filterDim.filters, filterDim.channels, filterDim.rows, filterDim.cols});
	this->_paramsHistory2[Bias]->reshape({filterDim.filters, 1, 1, 1});

	this->_paramsInitialized.resize(2);
	this->_paramsInitialized[Filter] = false;
	this->_paramsInitialized[Bias] = false;

    this->decayedBeta1 = 1.0;
    this->decayedBeta2 = 1.0;

	checkCUDNN(cudnnCreateTensorDescriptor(&this->inputTensorDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&this->outputTensorDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&this->biasTensorDesc));
	checkCUDNN(cudnnCreateFilterDescriptor(&this->filterDesc));
	checkCUDNN(cudnnCreateConvolutionDescriptor(&this->convDesc));

	checkCUDNN(cudnnSetTensor4dDescriptor(this->biasTensorDesc,
			CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, filterDim.filters, 1, 1));

    if (!deconv) {
	    checkCUDNN(cudnnSetFilter4dDescriptor(this->filterDesc,
			CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
			filterDim.filters, filterDim.channels, filterDim.rows, filterDim.cols));
    } else {
	    checkCUDNN(cudnnSetFilter4dDescriptor(this->filterDesc,
			CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
			filterDim.channels, filterDim.filters, filterDim.rows, filterDim.cols));
    }

	//int pad = (filterDim.rows-1)/2;
	checkCUDNN(cudnnSetConvolution2dDescriptor(this->convDesc,
			filterDim.pad, filterDim.pad, filterDim.stride, filterDim.stride, 1, 1,
			CUDNN_CROSS_CORRELATION,
			CUDNN_DATA_FLOAT));

	this->d_workspace = 0;
}

template <typename Dtype>
void ConvLayer<Dtype>::initialize(filter_dim filter_d, update_param weight_update_param,
    update_param bias_update_param, param_filler<Dtype> weight_filler,
    param_filler<Dtype> bias_filler, bool deconv, int deconvExtraCell) {

    if (!deconv)
	    this->type = Layer<Dtype>::Conv;
    else
	    this->type = Layer<Dtype>::Deconv;

    this->deconv = deconv;
    this->deconvExtraCell = deconvExtraCell;
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

	this->_paramsHistory2.resize(2);
	this->_paramsHistory2[Filter] = new Data<Dtype>(this->name + "_filter_history2");
	this->_paramsHistory2[Bias] = new Data<Dtype>(this->name + "_bias_history2");
	this->_paramsHistory2[Filter]->reshape(
        {filter_d.filters, filter_d.channels, filter_d.rows, filter_d.cols});
	this->_paramsHistory2[Bias]->reshape({filter_d.filters, 1, 1, 1});

	this->_paramsInitialized.resize(2);
	this->_paramsInitialized[Filter] = false;
	this->_paramsInitialized[Bias] = false;

    this->decayedBeta1 = 1.0;
    this->decayedBeta2 = 1.0;

	checkCUDNN(cudnnCreateTensorDescriptor(&inputTensorDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&outputTensorDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&biasTensorDesc));
	checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
	checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

	checkCUDNN(cudnnSetTensor4dDescriptor(biasTensorDesc,
			CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
			1, filter_d.filters, 1, 1));

    if (!this->deconv) {
	    checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc,
			CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
			filter_d.filters, filter_d.channels, filter_d.rows, filter_d.cols));
    } else {
	    checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc,
			CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
			filter_d.channels, filter_d.filters, filter_d.rows, filter_d.cols));
    }

	//int pad = (filter_d.rows-1)/2;
	checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc,
			filter_d.pad, filter_d.pad, filter_d.stride, filter_d.stride, 1, 1,
			CUDNN_CROSS_CORRELATION,
			CUDNN_DATA_FLOAT));

	//this->activation_fn = ActivationFactory<Dtype>::create(activationType);

	this->d_workspace = 0;
}

template <typename Dtype>
void ConvLayer<Dtype>::reshape() {
	Layer<Dtype>::_adjustInputShape();
	if (!Layer<Dtype>::_isInputShapeChanged(0))
		return;

	const bool deconv = SLPROP(Conv, deconv);
	const filter_dim& filterDim = SLPROP(Conv, filterDim);
	const int deconvExtraCell = SLPROP(Conv, deconvExtraCell);
	const string& name = SLPROP_BASE(name);

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
    if (!deconv) {
	    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(
			this->convDesc,
			this->inputTensorDesc,
			this->filterDesc,
			&n, &c, &h, &w));
    } else {
        SASSERT0(rows == cols);
        SASSERT0(filterDim.rows == filterDim.cols);
        SASSERT0(deconvExtraCell < filterDim.stride);

        // See ConvLayer.h
        n = batches;
        //c = filterDim.filters * filterDim.channels;
        c = filterDim.filters;
        h = filterDim.stride * (cols - 1) + filterDim.cols -
            2 * filterDim.pad + deconvExtraCell;
        w = filterDim.stride * (rows - 1) + filterDim.rows -
            2 * filterDim.pad + deconvExtraCell;
    }

    checkCUDNN(cudnnSetTensor4dDescriptor(
        this->outputTensorDesc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        n, c, h, w));

	const uint32_t obatches = static_cast<uint32_t>(n);
	const uint32_t ochannels = static_cast<uint32_t>(c);
	const uint32_t orows = static_cast<uint32_t>(h);
	const uint32_t ocols = static_cast<uint32_t>(w);

#if CONVLAYER_LOG
	printf("<%s> layer' output-0 has reshaped as: %dx%dx%dx%d\n",
			name.c_str(), obatches, ochannels, orows, ocols);
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

	const size_t memoryLimitInBytes = 8 << 20;
	//const size_t memoryLimitInBytes = 0;

	size_t convFwdWorkspaceSize;
	size_t convBwdFilterWorkspaceSize;
	size_t convBwdDataWorkspaceSize;

	// forward algorithm
    if (!deconv) {
	    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(
			Cuda::cudnnHandle,
			this->inputTensorDesc,
			this->filterDesc,
			this->convDesc,
			this->outputTensorDesc,
			//CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
			CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
			memoryLimitInBytes,
			&this->convFwdAlgo));

		checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
			Cuda::cudnnHandle,
			this->inputTensorDesc,
			this->filterDesc,
			this->convDesc,
			this->outputTensorDesc,
			this->convFwdAlgo,
			&convFwdWorkspaceSize));
    } else {
	    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(
			Cuda::cudnnHandle,
			this->outputTensorDesc,
			this->filterDesc,
			this->convDesc,
			this->inputTensorDesc,
			//CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
			CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
			8<<20,
			&this->convFwdAlgo));

		checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
			Cuda::cudnnHandle,
			this->outputTensorDesc,
			this->filterDesc,
			this->convDesc,
			this->inputTensorDesc,
			this->convFwdAlgo,
			&convFwdWorkspaceSize));
    }

	// backward filter algorithm
    if (!deconv) {
	    checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(
			Cuda::cudnnHandle,
			this->inputTensorDesc,
			this->outputTensorDesc,
			this->convDesc,
			this->filterDesc,
			//CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
			CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
			memoryLimitInBytes,
			&this->convBwdFilterAlgo));

	    checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
			Cuda::cudnnHandle,
			this->inputTensorDesc,
			this->outputTensorDesc,
			this->convDesc,
			this->filterDesc,
			this->convBwdFilterAlgo,
			&convBwdFilterWorkspaceSize));
    } else {
	    checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(
			Cuda::cudnnHandle,
			this->outputTensorDesc,
			this->inputTensorDesc,
			this->convDesc,
			this->filterDesc,
			//CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
			CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
			8<<20,
			&this->convBwdFilterAlgo));

	    checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
			Cuda::cudnnHandle,
			this->outputTensorDesc,
			this->inputTensorDesc,
			this->convDesc,
			this->filterDesc,
			this->convBwdFilterAlgo,
			&convBwdFilterWorkspaceSize));
    }

	// backward data algorithm
    if (!deconv) {
	    checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(
			Cuda::cudnnHandle,
			this->filterDesc,
			this->outputTensorDesc,
			this->convDesc,
			this->inputTensorDesc,
			//CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
			CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
			memoryLimitInBytes,
			&this->convBwdDataAlgo));

	    checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
			Cuda::cudnnHandle,
			this->filterDesc,
			this->outputTensorDesc,
			this->convDesc,
			this->inputTensorDesc,
			this->convBwdDataAlgo,
			&convBwdDataWorkspaceSize));
    } else {
	    checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(
			Cuda::cudnnHandle,
			this->filterDesc,
			this->inputTensorDesc,
			this->convDesc,
			this->outputTensorDesc,
			//CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
			CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
			8<<20,
			&this->convBwdDataAlgo));

	    checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
			Cuda::cudnnHandle,
			this->filterDesc,
			this->inputTensorDesc,
			this->convDesc,
			this->outputTensorDesc,
			this->convBwdDataAlgo,
			&convBwdDataWorkspaceSize));
    }

	this->workspaceSize = 0;
	this->workspaceSize = max(this->workspaceSize, convFwdWorkspaceSize);
	this->workspaceSize = max(this->workspaceSize, convBwdFilterWorkspaceSize);
	this->workspaceSize = max(this->workspaceSize, convBwdDataWorkspaceSize);


	if (this->workspaceSize > 0) {
		cout << name << "'s workspace: " << this->workspaceSize << endl;
		if (this->d_workspace) {
			checkCudaErrors(cudaFree(this->d_workspace));
			this->d_workspace = 0;
		}
		checkCudaErrors(Util::ucudaMalloc(&this->d_workspace, this->workspaceSize));
	}
}

template <typename Dtype>
void ConvLayer<Dtype>::update() {
	const filter_dim& filterDim = SLPROP(Conv, filterDim);
	const update_param& weightUpdateParam = SLPROP(Conv, weightUpdateParam);
	const update_param& biasUpdateParam = SLPROP(Conv, biasUpdateParam);
	const float weightDecay = SNPROP(weightDecay);
	const float epsilon = SNPROP(epsilon);
	const float decayRate = SNPROP(decayRate);
	const float beta1 = SNPROP(beta1);
	const float beta2 = SNPROP(beta2);


	// update filters ...
	const uint32_t weightSize = filterDim.size();
	const Dtype regScale = weightDecay * weightUpdateParam.decay_mult;
	const Dtype learnScale = 
			NetworkConfig<Dtype>::calcLearningRate() * weightUpdateParam.lr_mult;
    this->decayedBeta1 *= beta1;
    this->decayedBeta2 *= beta2;

	_updateParam(weightSize, regScale, learnScale, epsilon, decayRate, beta1, beta2, 
		this->_paramsHistory[Filter], this->_paramsHistory2[Filter], this->_params[Filter]);

	// update biases ...
	const uint32_t biasSize = filterDim.filters;
	const Dtype regScale_b = weightDecay * biasUpdateParam.decay_mult;
	const Dtype learnScale_b = 
			NetworkConfig<Dtype>::calcLearningRate() * biasUpdateParam.lr_mult;

	_updateParam(biasSize, regScale_b, learnScale_b, epsilon, decayRate, beta1, beta2, 
		this->_paramsHistory[Bias], this->_paramsHistory2[Bias], this->_params[Bias]);
}

template <typename Dtype>
void ConvLayer<Dtype>::_updateParam(const uint32_t paramSize, const Dtype regScale,
    const Dtype learnScale, const Dtype epsilon, const Dtype decayRate, const Dtype beta1, 
    const Dtype beta2, Data<Dtype>* dataHistory, Data<Dtype>* dataHistory2,
    Data<Dtype>* data) {
	const uint32_t batches = this->_inputData[0]->getShape(0);
	const Dtype normScale = 1.0/batches;
	const Dtype momentum = SNPROP(momentum);
	const Dtype negativeOne = -1.0;
	const Optimizer optimizer = (Optimizer)SNPROP(optimizer);

    if (!Worker<Dtype>::isSingle())
        data->mutable_host_grad();

	Dtype* d_paramGrad = data->mutable_device_grad();   // should update grad
	Dtype* d_paramData = data->mutable_device_data();
	Dtype* d_paramHistoryData = dataHistory->mutable_device_data();
	Dtype* d_paramHistoryData2 = dataHistory2->mutable_device_data();


    // FIXME: FullyConnectedLayer에 동일한 코드가 있음. 추후에 정리 필요
    // (1) do normalization & regularization
    //  FIXME: 이것도 옵션으로 정규화를 할지 여부를 설정할 수 있었으면 좋겠음.
#if 0
    checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(paramSize),
        &regScale, d_paramData, 1, d_paramGrad, 1));	// regularize
#endif

    // (2) apply optimizer
    if (optimizer == Optimizer::Momentum) {
        /****
         * Momentum Alogorithm
         *
         * v = mu * v - learning_rate * dx
         * x += v
         *
         */

    	soooa_gpu_axpy(static_cast<int>(paramSize), regScale, d_paramData, d_paramGrad);
		soooa_gpu_axpby(static_cast<int>(paramSize), learnScale, d_paramGrad, momentum,
			d_paramHistoryData);
		soooa_copy(static_cast<int>(paramSize), d_paramHistoryData, d_paramGrad);
		// update
		soooa_gpu_axpy(static_cast<int>(paramSize), negativeOne, d_paramGrad, d_paramData);
    } else if (optimizer == Optimizer::Vanilla) {
        /****
         * Vanilla Alogorithm
         *
         * x += -learning_rate * dx
         *
         */
    	checkCudaErrors(cublasSscal(Cuda::cublasHandle, static_cast<int>(paramSize),
            &learnScale, d_paramGrad, 1));				//
    	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(paramSize),
            &negativeOne, d_paramGrad, 1, d_paramData, 1));		// update
    } else if (optimizer == Optimizer::Nesterov) {
        /****
         * Nesterov Alogorithm
         *
         * v_prev = v # back this up
         * v = mu * v - learning_rate * dx # velocity update stays the same
         * x += -mu * v_prev + (1 + mu) * v # position update changes form
         *
         */
	    DoNesterov<<<SOOOA_GET_BLOCKS(static_cast<int>(paramSize)), SOOOA_CUDA_NUM_THREADS>>>(
            static_cast<int>(paramSize), d_paramGrad, d_paramHistoryData,
            d_paramHistoryData2, d_paramData, momentum, learnScale);
    } else if (optimizer == Optimizer::Adagrad) {
        /****
         * Adagrad Alogorithm
         *
         * cache += dx**2
         * x += -learning_rate * dx / (sqrt(cache) + eps)
         *
         */
	    DoAdagrad<<<SOOOA_GET_BLOCKS(static_cast<int>(paramSize)), SOOOA_CUDA_NUM_THREADS>>>(
            static_cast<int>(paramSize), d_paramGrad, d_paramHistoryData,
            d_paramData, learnScale, epsilon);

    } else if (optimizer == Optimizer::RMSprop) {
        /****
         * RMSprop
         *
         * cache = decay_rate * cache + (1 - decay_rate) * dx**2
         * x += - learning_rate * dx / (sqrt(cache) + eps)
         *
         */
	    DoRMSprop<<<SOOOA_GET_BLOCKS(static_cast<int>(paramSize)), SOOOA_CUDA_NUM_THREADS>>>(
            static_cast<int>(paramSize), d_paramGrad, d_paramHistoryData,
            d_paramData, learnScale, epsilon, decayRate);

    } else if (optimizer == Optimizer::Adam) {
        /****
         * Adam
         *
         * m = beta1 * m + (1 - beta1) * dx
         * v = beta2 * v + (1 - beta2) * (dx**2)
         * x += -learning_rate * m / (sqrt(v) + eps)
         *
         */
	    DoAdam<<<SOOOA_GET_BLOCKS(static_cast<int>(paramSize)), SOOOA_CUDA_NUM_THREADS>>>(
            static_cast<int>(paramSize), d_paramGrad, d_paramHistoryData, d_paramHistoryData2,
            d_paramData, learnScale, epsilon, beta1, beta2, this->decayedBeta1,
            this->decayedBeta2);
    } else {
        SASSERT(false, "invalid optimizer. optimizer=%d", (int)optimizer);
    }
}

template <typename Dtype>
void ConvLayer<Dtype>::applyChanges(LearnableLayer<Dtype> *targetLayer) {
	const filter_dim& filterDim = SLPROP(Conv, filterDim);
	const uint32_t weightSize = filterDim.size();
	const uint32_t biasSize = filterDim.filters;
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
	const filter_dim& filterDim = SLPROP(Conv, filterDim);
	const uint32_t weightSize = filterDim.size();
	const uint32_t biasSize = filterDim.filters;
    ConvLayer<Dtype>* _targetLayer = (ConvLayer<Dtype>*)targetLayer;

    memcpy(this->_params[Filter]->mutable_host_grad(), _targetLayer->_params[Filter]->host_grad(),
        weightSize);
    memcpy(this->_params[Bias]->mutable_host_grad(), _targetLayer->_params[Bias]->host_grad(),
        biasSize);
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

	const bool deconv = SLPROP(Conv, deconv);
    if (!deconv) {
	    checkCUDNN(cudnnConvolutionForward(Cuda::cudnnHandle,
			&Cuda::alpha, this->inputTensorDesc, d_inputData, this->filterDesc, d_filtersData,
            this->convDesc, this->convFwdAlgo, this->d_workspace, this->workspaceSize,
			&Cuda::beta, this->outputTensorDesc, d_outputData));
    } else {
	    checkCUDNN(cudnnConvolutionBackwardData(Cuda::cudnnHandle,
			&Cuda::alpha, this->filterDesc, d_filtersData, this->inputTensorDesc,
            d_inputData,
            this->convDesc, this->convBwdDataAlgo, this->d_workspace, this->workspaceSize,
			&Cuda::beta, this->outputTensorDesc, d_outputData));
    }

	const Dtype* d_biasesData = this->_params[Bias]->device_data();
	checkCUDNN(cudnnAddTensor(Cuda::cudnnHandle,
			&Cuda::alpha, this->biasTensorDesc, d_biasesData,
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
#endif

	// d(Cost)/d(Filters)
	const Dtype* d_inputData = this->_inputData[0]->device_data();
	//const Dtype* d_preActivationGrad = this->_preActivation->device_grad();
	const Dtype* d_outputGrad = this->_outputData[0]->device_grad();
	Dtype* d_filtersGrad = this->_params[Filter]->mutable_device_grad();

	const bool deconv = SLPROP(Conv, deconv);
    if (!deconv) {
	    checkCUDNN(cudnnConvolutionBackwardFilter(Cuda::cudnnHandle,
			&Cuda::alpha, this->inputTensorDesc, d_inputData, this->outputTensorDesc,
            d_outputGrad, this->convDesc, this->convBwdFilterAlgo, this->d_workspace, this->workspaceSize,
			&Cuda::beta, this->filterDesc, d_filtersGrad));
    } else {
	    checkCUDNN(cudnnConvolutionBackwardFilter(Cuda::cudnnHandle,
			&Cuda::alpha, this->outputTensorDesc, d_outputGrad, this->inputTensorDesc,
            d_inputData, this->convDesc, this->convBwdFilterAlgo, this->d_workspace, this->workspaceSize,
			&Cuda::beta, this->filterDesc, d_filtersGrad));
    }

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
			&Cuda::beta, this->biasTensorDesc, d_biasGrad));


}

template <typename Dtype>
void ConvLayer<Dtype>::_computeInputGrad() {
	// d(Cost)/d(Input)
	const Dtype* d_filtersData = this->_params[Filter]->device_data();
	//const Dtype* d_preActivationGrad = this->_preActivation->device_grad();
	const Dtype* d_outputGrad = this->_outputData[0]->device_grad();
    Dtype* d_inputGrad = this->_inputData[0]->mutable_device_grad();

    const bool deconv = SLPROP(Conv, deconv);
    if (!deconv) {
        checkCUDNN(cudnnConvolutionBackwardData(Cuda::cudnnHandle,
			&Cuda::alpha, this->filterDesc, d_filtersData, this->outputTensorDesc,
            d_outputGrad, this->convDesc, this->convBwdDataAlgo, this->d_workspace, this->workspaceSize,
			&Cuda::beta, this->inputTensorDesc, d_inputGrad));
    } else {
	    checkCUDNN(cudnnConvolutionForward(Cuda::cudnnHandle,
			&Cuda::alpha, this->outputTensorDesc, d_outputGrad, this->filterDesc, d_filtersData,
            this->convDesc, this->convFwdAlgo, this->d_workspace, this->workspaceSize,
			&Cuda::beta, this->inputTensorDesc, d_inputGrad));
    }

#if CONVLAYER_LOG
	this->_inputData[0]->print_grad("inputGrad:");
	this->_params[Filter]->print_data("filtersData:");
#endif
}

template void ConvLayer<float>::initialize();
template ConvLayer<float>::~ConvLayer();
template void ConvLayer<float>::initialize(filter_dim filter_d,
    update_param weight_update_param, update_param bias_update_param,
    param_filler<float> weight_filler, param_filler<float> bias_filler, bool deconv,
    int deconvExtraCell);
template void ConvLayer<float>::reshape();
template void ConvLayer<float>::update();
template void ConvLayer<float>::feedforward();
template void ConvLayer<float>::backpropagation();

/****************************************************************************
 * layer callback functions 
 ****************************************************************************/
template<typename Dtype>
void* ConvLayer<Dtype>::initLayer() {
    ConvLayer* conv = new ConvLayer<Dtype>(SLPROP(Conv, name));
    return (void*)conv;
}

template<typename Dtype>
void ConvLayer<Dtype>::destroyLayer(void* instancePtr) {
    ConvLayer<Dtype>* conv = (ConvLayer<Dtype>*)instancePtr;
    delete conv;
}

template<typename Dtype>
void ConvLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
    SASSERT0(index == 0);

    ConvLayer<Dtype>* conv = (ConvLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(conv->_inputData.size() == 0);
        conv->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(conv->_outputData.size() == 0);
        conv->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool ConvLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    ConvLayer<Dtype>* conv = (ConvLayer<Dtype>*)instancePtr;
    //conv->reshape();
    return true;
}

template<typename Dtype>
void ConvLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
    cout << "ConvLayer.. forward(). miniBatchIndex : " << miniBatchIdx << endl;
}

template<typename Dtype>
void ConvLayer<Dtype>::backwardTensor(void* instancePtr) {
    cout << "ConvLayer.. backward()" << endl;
}

template<typename Dtype>
void ConvLayer<Dtype>::learnTensor(void* instancePtr) {
    cout << "ConvLayer.. learn()" << endl;
}

template void* ConvLayer<float>::initLayer();
template void ConvLayer<float>::destroyLayer(void* instancePtr);
template void ConvLayer<float>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index);
template bool ConvLayer<float>::allocLayerTensors(void* instancePtr);
template void ConvLayer<float>::forwardTensor(void* instancePtr, int miniBatchIdx);
template void ConvLayer<float>::backwardTensor(void* instancePtr);
template void ConvLayer<float>::learnTensor(void* instancePtr);

#endif
