/*
 * FullyConnectedLayer.cpp
 *
 *  Created on: 2016. 5. 10.
 *      Author: jhkim
 */

#include "cuda_runtime.h"
#include <algorithm>

#include "FullyConnectedLayer.h"
#include "MathFunctions.h"
#include "Util.h"
#include "Network.h"
#include "SysLog.h"
#include "StdOutLog.h"
#include "PropMgmt.h"
#include "Update.h"
#include "Updater.h"
#include "Donator.h"

#define FULLYCONNECTEDLAYER_LOG 0

using namespace std;

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
__global__ void AddData(Dtype* dst, const Dtype* src, int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= N)
		return;

	dst[idx] = dst[idx] + src[idx]; 
}

template <typename Dtype>
FullyConnectedLayer<Dtype>::~FullyConnectedLayer() {
    if (SLPROP(FullyConnected, receive)) {
        Donator<Dtype>::releaseReceiver(SLPROP(FullyConnected, donatorID));
    } else {
        Util::clearVector(this->_params);
        Util::clearVector(this->_paramsHistory);
        Util::clearVector(this->_paramsHistory2);
    }
	checkCudaErrors(cudaFree(this->d_onevec));
}


template <typename Dtype>
void FullyConnectedLayer<Dtype>::reshape() {
	if (!Layer<Dtype>::_adjustInputShape()) {
		const uint32_t count = Util::vecCountByAxis(this->_inputShape[0], 1);
		const uint32_t inputDataCount = this->_inputData[0]->getCountByAxis(1);
		SASSERT0(count == inputDataCount);
	}

	/*
	// 배치수가 변경되는 경우는 허용하도록 하자.
	const uint32_t count = Util::vecCountByAxis(this->_inputShape[0], 1);
	const uint32_t inputDataCount = this->_inputData[0]->getCountByAxis(1);
	if (inputDataCount == count)
		return;
		*/

	// XXX: 주의

    // 여기에서는 batch 개수만 변경이 될 수 있다고 가정하였다.
    // 따라서 batch 개수에 대한 변경만 체크한다.
	if (!Layer<Dtype>::_isInputShapeChanged(0))
		return;

	this->batches = this->_inputData[0]->getShape(0);
	this->in_rows = this->_inputData[0]->getCountByAxis(SLPROP(FullyConnected, axis));
	this->out_rows = SLPROP(FullyConnected, nOut);

	const uint32_t channels = 1;
	const uint32_t cols = 1;

	//this->_inputShape[0] = {batches, channels, in_rows, cols};
	this->_inputShape[0] = this->_inputData[0]->getShape();
	this->_outputData[0]->reshape({this->batches, channels, this->out_rows, cols});

	/*
	checkCUDNN(cudnnSetTensor4dDescriptor(
			this->inputTensorDesc,
			CUDNN_TENSOR_NCHW,
			CUDNN_DATA_FLOAT,
			this->batches, channels, this->in_rows, cols));

	checkCUDNN(cudnnSetTensor4dDescriptor(
			this->outputTensorDesc,
			CUDNN_TENSOR_NCHW,
			CUDNN_DATA_FLOAT,
			this->batches, channels, this->out_rows, cols));
			*/

	STDOUT_COND_LOG(FULLYCONNECTEDLAYER_LOG, 
        "<%s> layer' input-0 has reshaped as: %dx%dx%dx%d\n",
        SLPROP_BASE(name).c_str(), this->batches, channels, this->in_rows, cols);
	STDOUT_COND_LOG(FULLYCONNECTEDLAYER_LOG,
	    "<%s> layer' output-0 has reshaped as: %dx%dx%dx%d\n", 
        SLPROP_BASE(name).c_str(), this->batches, channels, this->out_rows, cols);

	const uint32_t u_in = in_rows;
	const uint32_t u_out = out_rows;
	const uint32_t b_in = batches * in_rows;
	const uint32_t b_out = batches * out_rows;

	this->_params[ParamType::Weight]->reshape({1, 1, u_out, u_in});
	this->_params[ParamType::Bias]->reshape({1, u_out, 1, 1});
	this->_paramsHistory[ParamType::Weight]->reshape({1, 1, u_out, u_in});
	this->_paramsHistory[ParamType::Bias]->reshape({1, u_out, 1, 1});;
	this->_paramsHistory2[ParamType::Weight]->reshape({1, 1, u_out, u_in});
	this->_paramsHistory2[ParamType::Bias]->reshape({1, u_out, 1, 1});


	if (!this->_paramsInitialized[Weight]) {
        SLPROP(FullyConnected, weightFiller).fill(this->_params[ParamType::Weight]);
		this->_paramsInitialized[Weight] = true;
	}
	if (!this->_paramsInitialized[Bias]) {
        SLPROP(FullyConnected, weightFiller).fill(this->_params[ParamType::Bias]);
		this->_paramsInitialized[Bias] = true;
	}

	checkCudaErrors(Util::ucudaMalloc(&this->d_onevec, sizeof(Dtype)*batches));
	//FillValues<<<RoundUp(batches, BW), BW>>>(this->d_onevec, batches, 1.0f);
	FillValues<<<SOOOA_GET_BLOCKS(batches), SOOOA_CUDA_NUM_THREADS>>>(
			this->d_onevec, batches, 1.0f);

	this->_mask.reshape(b_out);
}

template <typename Dtype>
void FullyConnectedLayer<Dtype>::update() {
	//const uint32_t in_rows = this->_inputShape[0][2];
	//const uint32_t out_rows = this->_outputData[0]->getShape(2);

	const uint32_t weightSize = this->in_rows * this->out_rows;
	const Dtype regScale =
        SNPROP(weightDecay) * SLPROP(FullyConnected, weightUpdateParam).decay_mult;
	const Dtype learnScale = Update<Dtype>::calcLearningRate() *
		SLPROP(FullyConnected, weightUpdateParam).lr_mult;

    const Dtype beta1 = SNPROP(beta1);
    const Dtype beta2 = SNPROP(beta2);

    SLPROP(FullyConnected, decayedBeta1) *= beta1;
    SLPROP(FullyConnected, decayedBeta2) *= beta2;

    UpdateContext contextWeight = 
        Update<Dtype>::makeContext(weightSize, regScale, learnScale);

	Updater::updateParam(Weight, contextWeight, (void*)this->_paramsHistory[Weight],
        (void*)this->_paramsHistory2[Weight], (void*)this->_params[Weight]);

	const uint32_t biasSize = out_rows;
	const Dtype regScale_b = 
        SNPROP(weightDecay) * SLPROP(FullyConnected, biasUpdateParam).decay_mult;
	const Dtype learnScale_b = Update<Dtype>::calcLearningRate() *
        SLPROP(FullyConnected, biasUpdateParam).lr_mult;

    UpdateContext contextBias = 
        Update<Dtype>::makeContext(biasSize, regScale_b, learnScale_b);

	Updater::updateParam(Bias, contextBias, (void*)this->_paramsHistory[Bias],
        (void*)this->_paramsHistory2[Bias], (void*)this->_params[Bias]);
}

template <typename Dtype>
void FullyConnectedLayer<Dtype>::applyChanges(LearnableLayer<Dtype> *targetLayer) {
	//const uint32_t in_rows = this->_inputShape[0][2];
	//const uint32_t out_rows = this->_outputData[0]->getShape(2);

    const uint32_t weightSize = this->in_rows * this->out_rows;
    const uint32_t biasSize = this->out_rows;
    FullyConnectedLayer<Dtype>* _targetLayer = (FullyConnectedLayer<Dtype>*)targetLayer;

    //int blockSize = BW;
    int blockSize = SOOOA_CUDA_NUM_THREADS;
    int gridSize;

    gridSize = (weightSize + blockSize -1) / blockSize;

    AddData<<<gridSize, blockSize>>>(
        _targetLayer->_params[Weight]->mutable_device_grad(),
        this->_params[Weight]->device_grad(), weightSize);

    gridSize = (biasSize + blockSize -1) / blockSize;

    AddData<<<gridSize, blockSize>>>(
        _targetLayer->_params[Bias]->mutable_device_grad(),
        this->_params[Bias]->device_grad(), biasSize);
}

template <typename Dtype>
void FullyConnectedLayer<Dtype>::syncParams(LearnableLayer<Dtype> *targetLayer) {
	//const uint32_t in_rows = this->_inputShape[0][2];
	//const uint32_t out_rows = this->_outputData[0]->getShape(2);

    const uint32_t weightSize = this->in_rows * this->out_rows;
    const uint32_t biasSize = this->out_rows;
    FullyConnectedLayer<Dtype>* _targetLayer = (FullyConnectedLayer<Dtype>*)targetLayer;

    memcpy(this->_params[Weight]->mutable_host_grad(), _targetLayer->_params[Weight]->host_grad(),
        weightSize);
    memcpy(this->_params[Bias]->mutable_host_grad(), _targetLayer->_params[Bias]->host_grad(),
        biasSize);
}

template <typename Dtype>
void FullyConnectedLayer<Dtype>::feedforward() {
	reshape();

	_computeWeightedData();
	_computeWeightBiasedData();
}


template <typename Dtype>
void FullyConnectedLayer<Dtype>::_computeWeightedData() {
	//const uint32_t batches = this->_inputShape[0][0];
	//const uint32_t in_rows = this->_inputShape[0][2];
	//const uint32_t out_rows = this->_outputData[0]->getShape(2);

	// Apply weight to input data
	const Dtype* d_weightData = this->_params[Weight]->device_data();
	const Dtype* d_inputData = this->_inputData[0]->device_data();
	//Dtype* d_preActivationData = _preActivation->mutable_device_data();
	Dtype* d_outputData = this->_outputData[0]->mutable_device_data();

    /**
     * [cublasSgemm() 함수 설명 (from cuBlas User Documentation)]
     *
     * cublasStatus_t cublasSgemm(cublasHandle_t handle, cublasOperation_t transa,
     *                            cublasOperation_t transb, int m, int n, int k, 
     *                            const float *alpha, const float *A, int * lda, 
     *                            const float *B, int ldb, const float *beta, float *C, 
     *                            int ldc)
     *
     * C = α op ( A ) op ( B ) + β C
     *
     * where α and β are scalars, and A , B and C are matrices stored in column-major format
     * with dimensions op ( A ) m × k , op ( B ) k × n and C m × n , respectively. Also, for
     * matrix A 
     *
     * op ( A ) = A if  transa == CUBLAS_OP_N A T if  transa == CUBLAS_OP_T A H if  transa ==
     * CUBLAS_OP_C
     *
     * and op ( B ) is defined similarly for matrix B .
     *
     * cublasOperation_t option
     *  (1) CUBLAS_OP_N => the non-transpose operation is selected.
     *  (2) CUBLAS_OP_T => the transpose operation is selected.
     *  (3) CUBLAS_OP_C => the conjugate transpose operation is selected.
     *
     * lda,ldb,ldc => leading dimension of two-dimensional array used to store the matrix A,
     *                B, C
     */

	if (this->batches == 1) {
		soooa_gpu_gemv(CblasNoTrans,
				this->out_rows, this->in_rows,
				Cuda::alpha, d_weightData, d_inputData,
				Cuda::beta, d_outputData);

	} else {
		soooa_gpu_gemm(CblasNoTrans, CblasTrans,
				this->batches, this->out_rows, this->in_rows,
				Cuda::alpha, d_inputData, d_weightData,
				Cuda::beta, d_outputData);
	}
	/*
	checkCudaErrors(cublasSgemm(Cuda::cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
			this->out_rows, this->batches, this->in_rows,
			&Cuda::alpha, d_weightData, this->out_rows, d_inputData, this->in_rows,
			&Cuda::beta, d_outputData, this->out_rows));
			*/
}

template <typename Dtype>
void FullyConnectedLayer<Dtype>::_computeWeightBiasedData() {
	// Add bias to weighted input data
	const Dtype* d_biasData = this->_params[Bias]->device_data();
	//Dtype* d_preActivationData = _preActivation->mutable_device_data();
	Dtype* d_outputData = this->_outputData[0]->mutable_device_data();

	this->_params[Bias]->print_data();

	if (this->batches == 1) {
		soooa_gpu_axpy(this->out_rows, 1.0f,  d_biasData, d_outputData);
	} else {
		soooa_gpu_gemm(CblasNoTrans, CblasNoTrans,
				this->batches, this->out_rows,	1,
				Cuda::alpha, this->d_onevec, d_biasData,
				Cuda::alpha, d_outputData);
	}

	/*
	checkCudaErrors(cublasSgemm(Cuda::cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
			this->out_rows, this->batches, 1,
			&Cuda::alpha,
			d_biasData, this->out_rows,
			this->d_onevec, 1,
			&Cuda::alpha,
			d_outputData, this->out_rows));
			*/

	this->_params[Bias]->print_data();
}


template <typename Dtype>
void FullyConnectedLayer<Dtype>::_dropoutForward() {
	const double pDropOut = SLPROP(FullyConnected, pDropOut);
	const NetworkStatus status = (NetworkStatus)SNPROP(status);
	// TODO skip when test
	if(status == NetworkStatus::Train && pDropOut < 1.0f) {
		//int b_out = this->out_dim.batchsize();
		int b_out = this->_outputData[0]->getCount();
		Dtype* h_mask_mem = this->_mask.mutable_host_mem();

		for(int i = 0; i < b_out; i++) {
			h_mask_mem[i] = ((rand()/(RAND_MAX+1.0) > pDropOut)?1:0);
		}

		const Dtype* d_mask_mem = this->_mask.device_mem();
		Dtype* d_outputData = this->_outputData[0]->mutable_device_data();

		Dropout<<<SOOOA_GET_BLOCKS(b_out), SOOOA_CUDA_NUM_THREADS>>>(
				b_out, d_outputData, d_mask_mem, 0, this->scale, d_outputData);
	}
}

template <typename Dtype>
void FullyConnectedLayer<Dtype>::backpropagation() {
	//_dropoutBackward();

    /*
     * 아래와 같은 simple한 network layer가 있다고 가정하자.
     *
     *               <<<< ith layer >>>>            <<<< i+1th layer >>>>
     *   .....    Xi    Wi     Ai     Fi       Yi (=Xi+1)   ........
     *                  Bi
     *   .....    O ---------  O  ------------  O            ........
     *                                                     dL/dYi is already computed
     *
     *  (※  Xi = i번째 layer의 input 값, Wi = i번째 layer의 weight, 
     *      Bi = i번째 layer의 bias 값,  Ai = i번째 layer의 중간 값
     *      Fi = i번째 layer의 activation function
     *      Yi = i번째 layer의 ouput 값, i+1 번째 layer의 input 값이기도 함
     *      L = loss, dL/dYi = i+1번째 layer에서 계산되었던 gradient 값)
     *
     *  gradient descent 방식으로 학습을 하기 위해서는 dL/dWi & dL/dBi가 필요하다.
     *  체인 룰에 의하여 아래와 같은 식으로 표현이 된다:
     *  (가) dYi/dWi = dL/dYi * dYi/dAi * dAi/dWi
     *  (나) dYi/dBi = dL/dYi * dYi/dAi * dAi/dBi
     *
     *  (가),(나)를 계산하기 위해서는 아래와 같이 4가지 계산이 필요하다.
     *
     *  (A) dL/dYi : i+1번째 layer의 backward 과정에서 _outputData[0]의 grad에 값을 저장해
     *                두었다.
     *
     *  (B) dYi/dAi : _computePreActivationGrad() 에서 dL/dYi * dYi/dAi의 계산을  수행 한다. 
     *                dL/dYi는 구해져 있기 때문에 Yi, Ai 값이 필요하다. 이 값들은 forward시에
     *                각각 _outputData[0]의 data와 _preActivation의 data에 저장이 되어 있다.
     *                activation function에 맞게 Yi, Ai, dL/dYi를 입력값으로 하여 dL/dYi * 
     *                dYi/dAi 값이 계산이 되고, 결과값은 this->_preActivation의 grad에 담는다.
     *
     *  (C) dAi/dWi : _computeWeightGrad()에서 (A), (B)의 결과를 조합하여 weight Grad를
     *               계산한다. dAi/dWi는 실제로 transpose Xi이기 때문에 GEMM 연산만 진행
     *               한다. 결과값은 _params[Weight]의 grad에 저장된다.
     *
     *  (D) dAi/dBi : (C)과정과 동일하다. _computeBiasGrad()에서 bias를 계산하고, 그 결과 값을
     *                _params[Bias]의 grad에 저장을 하는 것만 다르다.
     *
     *  마지막으로 i-1 layer에게 dL/dYi-1값을 전달해야 한다. 이 과정은 _computeInputGrad()
     *  에서 수행이 된다. 결과값을 _inputData의 grad에 저장한다. dL/dYi-1 = dL/dXi =
     *   dL/dAi * dAi/dXi가 된다. dL/dAi는 _preAcitvation의 grad에 저장이 되어 있고, dAi/dXi는
     *  Wi의 transpose 이기 때문에 계산가능하다.
     */
	_computeWeightGrad();
	_computeBiasGrad();
	_computeInputGrad();
}

template <typename Dtype>
void FullyConnectedLayer<Dtype>::_dropoutBackward() {
	const double pDropOut = SLPROP(FullyConnected, pDropOut);
	const NetworkStatus status = (NetworkStatus)SNPROP(status);
	if(status == NetworkStatus::Train && pDropOut < 1.0f) {
		const uint32_t batchSize = this->_inputData[0]->getCount();

		this->_outputData[0]->print_grad("outputGrad:");
		const Dtype* d_mask_mem = this->_mask.device_mem();
		Dtype* d_outputGrad = this->_outputData[0]->mutable_device_grad();

		Dropout<<<SOOOA_GET_BLOCKS(batchSize), SOOOA_CUDA_NUM_THREADS>>>(
				batchSize, d_outputGrad, d_mask_mem, 0, this->scale, d_outputGrad);

		//_mask.print("mask:");
		this->_outputData[0]->print_grad("outputGrad:");
	}
}


template <typename Dtype>
void FullyConnectedLayer<Dtype>::_computeWeightGrad() {
	// d(Cost)/d(Weight)
	const Dtype* d_outputGrad = this->_outputData[0]->device_grad();
	const Dtype* d_inputData = this->_inputData[0]->device_data();
	Dtype* d_weightGrad = this->_params[Weight]->mutable_device_grad();

	soooa_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
			this->out_rows, this->in_rows, this->batches,
			Cuda::alpha, d_outputGrad, d_inputData,
			Cuda::alpha, d_weightGrad);

	/*
	checkCudaErrors(cublasSgemm(Cuda::cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
			this->out_rows, this->in_rows, this->batches,
			&Cuda::alpha, d_outputGrad, this->out_rows, d_inputData, this->in_rows,
			&Cuda::beta, d_weightGrad, this->out_rows));
			*/
}

template <typename Dtype>
void FullyConnectedLayer<Dtype>::_computeBiasGrad() {
	//const uint32_t batches = this->_inputShape[0][0];
	//const uint32_t in_rows = this->_inputShape[0][2];
	//const uint32_t out_rows = this->_outputData[0]->getShape(2);

	// d(Cost)/d(Bias) (same as d_preActivationGrad)
	//const Dtype* d_preActivationGrad = this->_preActivation->device_grad();
	const Dtype* d_outputGrad = this->_outputData[0]->device_grad();
	Dtype* d_biasGrad = this->_params[Bias]->mutable_device_grad();

	soooa_gpu_gemv<Dtype>(CblasTrans,
			this->batches, this->out_rows,
			Cuda::alpha, d_outputGrad, this->d_onevec,
			Cuda::alpha, d_biasGrad);

	/*
	checkCudaErrors(cublasSgemv(Cuda::cublasHandle, CUBLAS_OP_N,
			this->out_rows, this->batches,
			&Cuda::alpha, d_outputGrad, this->out_rows, this->d_onevec, 1,
			&Cuda::beta, d_biasGrad, 1));
			*/
	this->_params[Bias]->print_grad("biasGrad:");
	this->_params[Weight]->print_data("weightData:");
	//_preActivation->print_grad("preActivationGrad");
}

template <typename Dtype>
void FullyConnectedLayer<Dtype>::_computeInputGrad() {
	//const uint32_t batches = this->_inputShape[0][0];
	//const uint32_t in_rows = this->_inputShape[0][2];
	//const uint32_t out_rows = this->_outputData[0]->getShape(2);

	// d(Cost)/d(Input)
	const Dtype* d_weightData = this->_params[Weight]->device_data();
	//const Dtype* d_preActivationGrad = this->_preActivation->device_grad();
	const Dtype* d_outputGrad = this->_outputData[0]->device_grad();
	Dtype* d_inputGrad = this->_inputData[0]->mutable_device_grad();

	soooa_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
			this->batches, this->in_rows, this->out_rows,
			Cuda::alpha, d_outputGrad, d_weightData,
			Cuda::beta, d_inputGrad);

	/*
	checkCudaErrors(cublasSgemm(Cuda::cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
			this->in_rows, this->batches, this->out_rows,
			&Cuda::alpha, d_weightData, this->out_rows, d_outputGrad, this->out_rows,
			&Cuda::beta, d_inputGrad, this->in_rows));
			*/
	this->_inputData[0]->print_grad("inputGrad:");

	/*
	if(this->_input->is_nan_grad()) {
		cout << SLPROP_BASE(name) << " _input gradient nan ... " << endl;
		Data<Dtype>::printConfig = 1;
		this->_input->print_grad("deltaInput:");
		Data<Dtype>::printConfig = 0;
		exit(1);
	}
	*/
}

template FullyConnectedLayer<float>::~FullyConnectedLayer();
template void FullyConnectedLayer<float>::reshape();
template void FullyConnectedLayer<float>::update();
template void FullyConnectedLayer<float>::feedforward();
template void FullyConnectedLayer<float>::backpropagation();


/*
template void* FullyConnectedLayer<float>::initLayer();
template void FullyConnectedLayer<float>::destroyLayer(void* instancePtr);
template void FullyConnectedLayer<float>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index);
template bool FullyConnectedLayer<float>::allocLayerTensors(void* instancePtr);
template void FullyConnectedLayer<float>::forwardTensor(void* instancePtr, int miniBatchIdx);
template void FullyConnectedLayer<float>::backwardTensor(void* instancePtr);
template void FullyConnectedLayer<float>::learnTensor(void* instancePtr);
*/
