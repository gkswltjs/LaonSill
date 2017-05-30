/*
 * SoftmaxWithLossLayer.cpp
 *
 *  Created on: Dec 3, 2016
 *      Author: jkim
 */

#include <cfloat>
#include <vector>

#include "SoftmaxWithLossLayer.h"
#include "common.h"
#include "MathFunctions.h"
#include "SysLog.h"
#include "PropMgmt.h"
#include "InnerLayerFunc.h"

#define SOFTMAXWITHLOSSLAYER_LOG 0

using namespace std;

template<typename Dtype>
SoftmaxWithLossLayer<Dtype>::SoftmaxWithLossLayer()
: LossLayer<Dtype>(),
  prob("prob") {
	this->type = Layer<Dtype>::SoftmaxWithLoss;

	const bool hasNormalization = SLPROP(Loss, hasNormalization);
	const bool hasNormalize = SLPROP(Loss, hasNormalize);
	const bool normalize = SLPROP(Loss, normalize);
	if (!hasNormalization && hasNormalize)
		SLPROP(SoftmaxWithLoss, normalization) = normalize ?
				LossLayer<Dtype>::NormalizationMode::Valid :
				LossLayer<Dtype>::NormalizationMode::BatchSize;

	InnerLayerFunc::initLayer(0);
}

template <typename Dtype>
SoftmaxWithLossLayer<Dtype>::~SoftmaxWithLossLayer() {
    InnerLayerFunc::destroyLayer(0);
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::reshape() {
	bool adjusted = Layer<Dtype>::_adjustInputShape();
	if (adjusted) {
		this->_outputData[0]->reshape({1, 1, 1, 1});
#if SOFTMAXWITHLOSSLAYER_LOG
		printf("<%s> layer' output-0 has reshaped as: %dx%dx%dx%d\n",
				SLPROP_BASE(name).c_str(), 1, 1, 1, 1);
#endif

        cout << "set inout tensor" << endl;
        InnerLayerFunc::setInOutTensor(0, (void*)this->_inputData[0], true, 0);
        InnerLayerFunc::setInOutTensor(0, (void*)&this->prob, false, 0);
	}

	const uint32_t inputSize = this->_inputData.size();
	for (uint32_t i = 0; i < inputSize; i++) {
		if (!Layer<Dtype>::_isInputShapeChanged(i))
			continue;

		const vector<uint32_t>& inputDataShape = this->_inputData[i]->getShape();
		this->_inputShape[i] = inputDataShape;

		// "data"
		if (i == 0) {
			this->prob.reshape(inputDataShape);

			//SLPROP(SoftmaxWithLoss, softmaxAxis) = 1;
			this->outerNum = this->_inputData[0]->getCountByAxis(0, 
                    SLPROP(SoftmaxWithLoss, softmaxAxis));
			this->innerNum = this->_inputData[0]->getCountByAxis(
                    SLPROP(SoftmaxWithLoss, softmaxAxis)+1);

            SASSERT(this->outerNum*this->innerNum == this->_inputData[1]->getCount(),
			    "Number of labels must match number of predictions ... "
                "outer num : %d, inner num : %d, input count : %lu",
                this->outerNum, this->innerNum, this->_inputData[1]->getCount());

#if 0
			assert(this->outerNum*this->innerNum == this->_inputData[1]->getCount() &&
					"Number of labels must match number of predictions ... ");
#endif

			if (this->_outputData.size() > 1) {
				// softmax output ...
			}
		}
		// "labels"
	}
}



template <typename Dtype>
__global__ void SoftmaxLossForwardGPU(
		const int nthreads,
        const Dtype* prob_data,
        const Dtype* label,
        Dtype* loss,
        const int num,
        const int dim,
        const int spatial_dim,
        const bool has_ignore_label_,
        const int ignore_label_,
        Dtype* counts) {

	CUDA_KERNEL_LOOP(index, nthreads) {
		//printf("SoftmaxLossForwardGPU index: %d\n", index);

		const int n = index / spatial_dim;
		const int s = index % spatial_dim;
		const int label_value = static_cast<int>(label[n * spatial_dim + s]);
		if (has_ignore_label_ && label_value == ignore_label_) {
			loss[index] = 0;
			counts[index] = 0;
		} else {
			loss[index] = -log(max(prob_data[n * dim + label_value * spatial_dim + s],
					Dtype(FLT_MIN)));
			counts[index] = 1;
		}
	}
}


template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::feedforward() {
	reshape();



	//Data<Dtype>::printConfig = true;
	//SyncMem<Dtype>::printConfig = true;

#if 0
	this->_inputData[0]->print_data({}, false);
	this->_inputData[1]->print_data({}, false);

	this->softmaxLayer->feedforward();

	this->softmaxLayer->_outputData[0]->print_data({}, false);
#else
    InnerLayerFunc::runForward(0, -1);
#endif


	const Dtype* probData = this->prob.device_data();
	const Dtype* label = this->_inputData[1]->device_data();
	const int dim = this->prob.getCount() / this->outerNum;
	const int nthreads = this->outerNum * this->innerNum;
	// Since this memory is not used for anything until it is overwritten
	// on the backward pass, we use it here to avoid having to allocate new GPU
	// memory to accumelate intermediate results in the kernel.
	Dtype* lossData = this->_inputData[0]->mutable_device_grad();
	// Similary, this memroy is never used elsewhere, and thus we can use it
	// to avoid having to allocated additional GPU memory.
	Dtype* counts = this->prob.mutable_device_grad();


	//cout << "FLT_MIN: " << Dtype(FLT_MIN) << endl;



	SoftmaxLossForwardGPU<Dtype><<<SOOOA_GET_BLOCKS(nthreads), SOOOA_CUDA_NUM_THREADS>>>(
			nthreads, probData, label, lossData, this->outerNum, dim,
			this->innerNum, SLPROP(SoftmaxWithLoss, hasIgnoreLabel), SLPROP(SoftmaxWithLoss, ignoreLabel), counts);
	//cudaDeviceSynchronize();


	//exit(1);

	/*
	if (SLPROP_BASE(name) == "loss_cls") {
		Data<Dtype>::printConfig = true;

		//this->_inputData[0]->print_data({}, false);
		//this->prob.print_data({}, false);
		this->_inputData[1]->print_data({}, false);
		//this->_inputData[0]->print_grad({}, false);
		//this->prob.print_grad({}, false);

		Data<Dtype>::printConfig = false;
	}
	*/


	this->_inputData[0]->print_grad({}, false);

	Dtype loss;
	soooa_gpu_asum(nthreads, lossData, &loss);
	Dtype validCount = -1;
	// Only launch another CUDA kernel if we actually need the count of valid
	// outputs.
	if (SLPROP(SoftmaxWithLoss, normalization) == LossLayer<Dtype>::NormalizationMode::Valid &&
			SLPROP(SoftmaxWithLoss, hasIgnoreLabel))
		soooa_gpu_asum(nthreads, counts, &validCount);

	// xxx normalizer test -> restored
	this->_outputData[0]->mutable_host_data()[0] = loss *
			Dtype(SLPROP(SoftmaxWithLoss, lossWeight)) / getNormalizer(validCount);
	//this->_outputData[0]->mutable_host_data()[0] = loss * Dtype(SLPROP(SoftmaxWithLoss, lossWeight));


	//cout << "softmaxwithloss: " << this->_outputData[0]->host_data()[0] << endl;

	// XXX: output data가 복수개인 경우 처리 ...
	//if (this->_outputData.size() == 2) {
	//
	//}


	//Data<Dtype>::printConfig = true;
	//this->prob.reshape({1, 2*9, this->prob.getShape(2)/9, this->prob.getShape(3)});
	//this->prob.print_data({}, false);
	//Data<Dtype>::printConfig = false;
	//exit(1);


	this->_outputData[0]->print_data({}, false);

	Data<Dtype>::printConfig = false;
	SyncMem<Dtype>::printConfig = false;
}




template <typename Dtype>
__global__ void SoftmaxLossBackwardGPU(
		const int nthreads,
		const Dtype* top,
        const Dtype* label,
        Dtype* bottom_diff,
        const int num,
        const int dim,
        const int spatial_dim,
        const bool has_ignore_label_,
        const int ignore_label_,
        Dtype* counts) {
	const int channels = dim / spatial_dim;
	CUDA_KERNEL_LOOP(index, nthreads) {
		const int n = index / spatial_dim;
		const int s = index % spatial_dim;
		const int label_value = static_cast<int>(label[n * spatial_dim + s]);

		if (has_ignore_label_ && label_value == ignore_label_) {
			for (int c = 0; c < channels; ++c) {
				bottom_diff[n * dim + c * spatial_dim + s] = 0;
			}
			counts[index] = 0;
		} else {
			bottom_diff[n * dim + label_value * spatial_dim + s] -= 1;
			counts[index] = 1;
		}
	}
}


template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::backpropagation() {
    SASSERT0(SLPROP_BASE(propDown)[1] == false);

	if (SLPROP_BASE(propDown)[0]) {
		Dtype* inputGrad = this->_inputData[0]->mutable_device_grad();
		const Dtype* probData = this->prob.device_data();
		const Dtype* outputData = this->_outputData[0]->device_data();
		soooa_gpu_memcpy(this->prob.getCount() * sizeof(Dtype), probData, inputGrad);
		const Dtype* label = this->_inputData[1]->device_data();
		const int dim = this->prob.getCount() / this->outerNum;
		const int nthreads = this->outerNum * this->innerNum;
		// Since this memroy is never used for anything else,
		// we use to avoid allocating new GPU memroy.
		Dtype* counts = this->prob.mutable_device_grad();

		SoftmaxLossBackwardGPU<Dtype><<<SOOOA_GET_BLOCKS(nthreads),
            SOOOA_CUDA_NUM_THREADS>>>(nthreads, outputData, label, inputGrad,
            this->outerNum, dim, this->innerNum, SLPROP(SoftmaxWithLoss, hasIgnoreLabel), SLPROP(SoftmaxWithLoss, ignoreLabel),
            counts);

		Dtype validCount = -1;
		// Only launch another CUDA kernel if we actually need the count of valid
		// outputs.
		if (SLPROP(SoftmaxWithLoss, normalization) == LossLayer<Dtype>::NormalizationMode::Valid &&
				SLPROP(SoftmaxWithLoss, hasIgnoreLabel))
			soooa_gpu_asum(nthreads, counts, &validCount);

		const Dtype lossWeight = Dtype(1) / getNormalizer(validCount);
		soooa_gpu_scal(this->prob.getCount(), lossWeight, inputGrad);




		/*
		Data<Dtype>::printConfig = true;
		SyncMem<Dtype>::printConfig = true;
		this->_outputData[0]->print_grad({}, false);
		this->_inputData[0]->print_grad({}, false);
		Data<Dtype>::printConfig = false;
		SyncMem<Dtype>::printConfig = false;
		*/

	}
}





template <typename Dtype>
Dtype SoftmaxWithLossLayer<Dtype>::cost() {
	return this->_outputData[0]->host_data()[0];
}

template <typename Dtype>
Dtype SoftmaxWithLossLayer<Dtype>::getNormalizer(int validCount) {
	Dtype normalizer;
	switch (SLPROP(SoftmaxWithLoss, normalization)) {
	case LossLayer<Dtype>::NormalizationMode::Full:
		normalizer = Dtype(this->outerNum * this->innerNum);
		break;
	case LossLayer<Dtype>::NormalizationMode::Valid:
		if (validCount == -1) {
			normalizer = Dtype(this->outerNum * this->innerNum);
		} else {
			normalizer = Dtype(validCount);
		}
		break;
	case LossLayer<Dtype>::NormalizationMode::BatchSize:
		normalizer = Dtype(this->outerNum);
		break;
	case LossLayer<Dtype>::NormalizationMode::NoNormalization:
		normalizer = Dtype(1);
		break;
	default:
		SASSERT(false, "Unknown normlization mode ... ");
	}
	// Some useres will have no labels for some examples in order to 'turn off' a
	// particular loss in a multi-task setup. The max prevents NaNs in that case.
	return max(Dtype(1.0), normalizer);
}

/****************************************************************************
 * layer callback functions 
 ****************************************************************************/
template<typename Dtype>
void* SoftmaxWithLossLayer<Dtype>::initLayer() {
    SoftmaxWithLossLayer* layer = new SoftmaxWithLossLayer<Dtype>();
    return (void*)layer;
}

template<typename Dtype>
void SoftmaxWithLossLayer<Dtype>::destroyLayer(void* instancePtr) {
    SoftmaxWithLossLayer<Dtype>* layer = (SoftmaxWithLossLayer<Dtype>*)instancePtr;
    delete layer;
}

template<typename Dtype>
void SoftmaxWithLossLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {

    SoftmaxWithLossLayer<Dtype>* layer = (SoftmaxWithLossLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(index < 2);
        SASSERT0(layer->_inputData.size() == index);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(index == 0);
        SASSERT0(layer->_outputData.size() == 0);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool SoftmaxWithLossLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    SoftmaxWithLossLayer<Dtype>* layer = (SoftmaxWithLossLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void SoftmaxWithLossLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
    SoftmaxWithLossLayer<Dtype>* layer = (SoftmaxWithLossLayer<Dtype>*)instancePtr;
    layer->feedforward();
}

template<typename Dtype>
void SoftmaxWithLossLayer<Dtype>::backwardTensor(void* instancePtr) {
    SoftmaxWithLossLayer<Dtype>* layer = (SoftmaxWithLossLayer<Dtype>*)instancePtr;
    layer->backpropagation();
}

template<typename Dtype>
void SoftmaxWithLossLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template class SoftmaxWithLossLayer<float>;
