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

#define SOFTMAXWITHLOSSLAYER_LOG 0

using namespace std;

template <typename Dtype>
SoftmaxWithLossLayer<Dtype>::SoftmaxWithLossLayer(Builder* builder)
	: LossLayer<Dtype>(builder) {
	this->softmaxAxis = builder->_softmaxAxis;

	initialize();
}

template <typename Dtype>
SoftmaxWithLossLayer<Dtype>::~SoftmaxWithLossLayer() {
	delete this->prob;
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::reshape() {
	bool adjusted = Layer<Dtype>::_adjustInputShape();
	if (adjusted) {
		this->_outputData[0]->reshape({1, 1, 1, 1});
		this->_outputData[0]->mutable_host_grad()[0] = this->lossWeight;

#if SOFTMAXWITHLOSSLAYER_LOG
		printf("<%s> layer' output-0 has reshaped as: %dx%dx%dx%d\n",
				this->name.c_str(), 1, 1, 1, 1);
#endif

		this->softmaxLayer->_inputData.push_back(this->_inputData[0]);
		this->softmaxLayer->_outputData.push_back(this->prob);
	}

	const uint32_t inputSize = this->_inputData.size();
	for (uint32_t i = 0; i < inputSize; i++) {
		if (!Layer<Dtype>::_isInputShapeChanged(i))
			continue;

		const vector<uint32_t>& inputDataShape = this->_inputData[i]->getShape();
		this->_inputShape[i] = inputDataShape;

		// "data"
		if (i == 0) {
			this->prob->reshape(inputDataShape);

			//this->softmaxAxis = 1;
			this->outerNum = this->_inputData[0]->getCountByAxis(0, this->softmaxAxis);
			this->innerNum = this->_inputData[0]->getCountByAxis(this->softmaxAxis+1);

			SASSERT(this->outerNum*this->innerNum == this->_inputData[1]->getCount(),
					"Number of labels must match number of predictions ... ");

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

	this->softmaxLayer->feedforward();


	const Dtype* probData = this->prob->device_data();
	const Dtype* label = this->_inputData[1]->device_data();
	const int dim = this->prob->getCount() / this->outerNum;
	const int nthreads = this->outerNum * this->innerNum;
	// Since this memory is not used for anything until it is overwritten
	// on the backward pass, we use it here to avoid having to allocate new GPU
	// memory to accumelate intermediate results in the kernel.
	Dtype* lossData = this->_inputData[0]->mutable_device_grad();
	// Similary, this memroy is never used elsewhere, and thus we can use it
	// to avoid having to allocated additional GPU memory.
	Dtype* counts = this->prob->mutable_device_grad();


	//cout << "FLT_MIN: " << Dtype(FLT_MIN) << endl;



	SoftmaxLossForwardGPU<Dtype><<<SOOOA_GET_BLOCKS(nthreads), SOOOA_CUDA_NUM_THREADS>>>(
			nthreads, probData, label, lossData, this->outerNum, dim,
			this->innerNum, this->hasIgnoreLabel, this->ignoreLabel, counts);
	CUDA_POST_KERNEL_CHECK;
	//cudaDeviceSynchronize();


	//exit(1);

	/*
	if (this->name == "loss_cls") {
		Data<Dtype>::printConfig = true;

		//this->_inputData[0]->print_data({}, false);
		//this->prob->print_data({}, false);
		this->_inputData[1]->print_data({}, false);
		//this->_inputData[0]->print_grad({}, false);
		//this->prob->print_grad({}, false);

		Data<Dtype>::printConfig = false;
	}
	*/

	Dtype loss;
	soooa_gpu_asum(nthreads, lossData, &loss);
	Dtype validCount = -1;
	// Only launch another CUDA kernel if we actually need the count of valid
	// outputs.
	if (this->normalization == LossLayer<Dtype>::NormalizationMode::Valid &&
			this->hasIgnoreLabel)
		soooa_gpu_asum(nthreads, counts, &validCount);

	// xxx normalizer test -> restored
	this->_outputData[0]->mutable_host_data()[0] = loss *
			Dtype(this->lossWeight) / LossLayer<Dtype>::getNormalizer(this->outerNum,
					this->innerNum, validCount);
	//this->_outputData[0]->mutable_host_data()[0] = loss * Dtype(this->lossWeight);


	//cout << "softmaxwithloss: " << this->_outputData[0]->host_data()[0] << endl;

	// XXX: output data가 복수개인 경우 처리 ...
	//if (this->_outputData.size() == 2) {
	//
	//}


	//Data<Dtype>::printConfig = true;
	//this->prob->reshape({1, 2*9, this->prob->getShape(2)/9, this->prob->getShape(3)});
	//this->prob->print_data({}, false);
	//Data<Dtype>::printConfig = false;
	//exit(1);

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
	assert(!this->_propDown[1] &&
			"SoftmaxLayer cannot backpropagate to label inputs ... ");

	if (this->_propDown[0]) {
		Dtype* inputGrad = this->_inputData[0]->mutable_device_grad();
		const Dtype* probData = this->prob->device_data();
		const Dtype* outputData = this->_outputData[0]->device_data();
		soooa_gpu_memcpy(this->prob->getCount() * sizeof(Dtype), probData, inputGrad);
		const Dtype* label = this->_inputData[1]->device_data();
		const int dim = this->prob->getCount() / this->outerNum;
		const int nthreads = this->outerNum * this->innerNum;
		// Since this memroy is never used for anything else,
		// we use to avoid allocating new GPU memroy.
		Dtype* counts = this->prob->mutable_device_grad();

		SoftmaxLossBackwardGPU<Dtype><<<SOOOA_GET_BLOCKS(nthreads),
            SOOOA_CUDA_NUM_THREADS>>>(nthreads, outputData, label, inputGrad,
            this->outerNum, dim, this->innerNum, this->hasIgnoreLabel, this->ignoreLabel,
            counts);
		CUDA_POST_KERNEL_CHECK;

		Dtype validCount = -1;
		// Only launch another CUDA kernel if we actually need the count of valid
		// outputs.
		if (this->normalization == LossLayer<Dtype>::NormalizationMode::Valid &&
				this->hasIgnoreLabel)
			soooa_gpu_asum(nthreads, counts, &validCount);

		const Dtype lossWeight = Dtype(1) / LossLayer<Dtype>::getNormalizer(this->outerNum,
				this->innerNum, validCount);
		soooa_gpu_scal(this->prob->getCount(), lossWeight, inputGrad);
	}
}






template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::initialize() {
	this->type = Layer<Dtype>::SoftmaxWithLoss;


	//assert(this->hasNormalize);
	if (!this->hasNormalization &&
			this->hasNormalize)
		this->normalization = this->normalize ?
				LossLayer<Dtype>::NormalizationMode::Valid :
				LossLayer<Dtype>::NormalizationMode::BatchSize;
	//else
	//	this->normalization =


	this->prob = new Data<Dtype>("prob");

	// XXX: float로 생성하지 않으니 error ...
	// create inner softmax layer
	SoftmaxLayer<float>::Builder* softmaxLayerBuilder =
			new typename SoftmaxLayer<float>::Builder();

	softmaxLayerBuilder
		->id(0)
		->name("inner_softmax")
		->softmaxAxis(this->softmaxAxis)
		->inputs({"inner_softmax_10_input"})
		->outputs({"inner_softmax_10_input"});
	this->softmaxLayer = dynamic_cast<SoftmaxLayer<Dtype>*>(softmaxLayerBuilder->build());

}

template <typename Dtype>
Dtype SoftmaxWithLossLayer<Dtype>::cost() {
	return this->_outputData[0]->host_data()[0];
}

/*
template <typename Dtype>
Dtype SoftmaxWithLossLayer<Dtype>::getNormalizer(int validCount) {
	Dtype normalizer;
	switch (this->normalization) {
	case LossLayer<Dtype>::NormalizationMode::Full:
		normalizer = Dtype(outerNum * innerNum);
		break;
	case LossLayer<Dtype>::NormalizationMode::Valid:
		if (validCount == -1) {
			normalizer = Dtype(outerNum * innerNum);
		} else {
			normalizer = Dtype(validCount);
		}
		break;
	case LossLayer<Dtype>::NormalizationMode::BatchSize:
		normalizer = Dtype(outerNum);
		break;
	case LossLayer<Dtype>::NormalizationMode::NoNormalization:
		normalizer = Dtype(1);
		break;
	default:
		cout << "Unknown normlization mode ... " << endl;
		exit(-1);
	}
	// Some useres will have no labels for some examples in order to 'turn off' a
	// particular loss in a multi-task setup. The max prevents NaNs in that case.
	return max(Dtype(1.0), normalizer);
}
*/


template class SoftmaxWithLossLayer<float>;
