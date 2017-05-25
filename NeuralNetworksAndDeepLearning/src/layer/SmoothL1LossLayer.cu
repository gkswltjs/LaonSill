/*
 * SmoothL1LossLayer.cpp
 *
 *  Created on: Nov 23, 2016
 *      Author: jkim
 */

#include <vector>

#include "SmoothL1LossLayer.h"
#include "MathFunctions.h"
#include "PropMgmt.h"
#include "SysLog.h"

#define SMOOTHL1LOSSLAYER_LOG 0

using namespace std;


template <typename Dtype>
SmoothL1LossLayer<Dtype>::SmoothL1LossLayer(const string& name)
: LossLayer<Dtype>(name),
  diff("diff"),
  errors("errors"),
  ones("ones") {
	initialize();
}

template <typename Dtype>
SmoothL1LossLayer<Dtype>::SmoothL1LossLayer(Builder* builder)
: LossLayer<Dtype>(builder),
  diff("diff"),
  errors("errors"),
  ones("ones") {
	this->sigma2 = builder->_sigma * builder->_sigma;
	this->firstAxis = builder->_firstAxis;
	initialize();
}

template <typename Dtype>
SmoothL1LossLayer<Dtype>::~SmoothL1LossLayer() {
}

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::reshape() {
	bool adjusted = Layer<Dtype>::_adjustInputShape();
	if (adjusted) {
		this->hasWeights = (this->_inputData.size() >= 3);
		SASSERT(this->hasWeights && this->_inputData.size() == 4,
				"If weights are used, must specify both inside and outside weights");

		this->_outputData[0]->reshape({1, 1, 1, 1});
#if SMOOTHL1LOSSLAYER_LOG
		printf("<%s> layer' output-0 has reshaped as: %dx%dx%dx%d\n",
				this->name.c_str(), 1, 1, 1, 1);
#endif
	}

	const uint32_t inputSize = this->_inputData.size();
	for (uint32_t i = 0; i < inputSize; i++) {
		if (!Layer<Dtype>::_isInputShapeChanged(i))
			continue;

		const vector<uint32_t>& inputDataShape = this->_inputData[i]->getShape();
		this->_inputShape[i] = inputDataShape;

		// rpn_bbox_pred
		if (i == 0) {
			this->diff.reshape(inputDataShape);
			this->errors.reshape(inputDataShape);
			// vector of ones used to sum
			this->ones.reshape(inputDataShape);
			this->ones.reset_host_data(false, 1.0f);

		}
		// rpn_bbox_targets
		else if (i == 1) {
			SASSERT0(this->_inputData[0]->channels() == this->_inputData[1]->channels());
			SASSERT0(this->_inputData[0]->height() == this->_inputData[1]->height());
			SASSERT0(this->_inputData[0]->width() == this->_inputData[1]->width());
		}
		// rpn_bbox_inside_weights
		else if (i == 2) {
			if (this->hasWeights) {
				SASSERT0(this->_inputData[0]->channels() == this->_inputData[2]->channels());
				SASSERT0(this->_inputData[0]->height() == this->_inputData[2]->height());
				SASSERT0(this->_inputData[0]->width() == this->_inputData[2]->width());
			}
		}
		// rpn_bbox_outside_weights
		else if (i == 3) {
			if (this->hasWeights) {
				SASSERT0(this->_inputData[0]->channels() == this->_inputData[3]->channels());
				SASSERT0(this->_inputData[0]->height() == this->_inputData[3]->height());
				SASSERT0(this->_inputData[0]->width() == this->_inputData[3]->width());
			}
		}
	}
}


template <typename Dtype>
__global__ void SmoothL1Forward(const uint32_t n, const Dtype* in, Dtype* out,
    Dtype sigma2) {
  // f(x) = 0.5 * (sigma2 * x)^2          if |x| < 1 / sigma2 / sigma2
  //        |x| - 0.5 / sigma2 / sigma2    otherwise
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = in[index];
    Dtype abs_val = abs(val);
    if (abs_val < 1.0 / sigma2) {
      out[index] = 0.5 * val * val * sigma2;
    } else {
      out[index] = abs_val - 0.5 / sigma2;
    }
  }
}

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::feedforward() {
	reshape();

	const uint32_t count = this->_inputData[0]->getCount();
	// prediction (inputData[0]) - target (inputData[1]) => diff
	soooa_gpu_sub(
			count,
			this->_inputData[0]->device_data(),
			this->_inputData[1]->device_data(),
			this->diff.mutable_device_data());		// d := b0 - b1


#if SMOOTHL1LOSSLAYER_LOG
	Data<Dtype>::printConfig = true;
	Data<Dtype>::printConfig = false;
#endif

#if SMOOTHL1LOSSLAYER_LOG
	Data<Dtype>::printConfig = true;
	this->_inputData[0]->print_data();
	this->_inputData[1]->print_data();
	this->diff.print_data();
	Data<Dtype>::printConfig = false;
#endif

	if (this->hasWeights) {

#if SMOOTHL1LOSSLAYER_LOG
	Data<Dtype>::printConfig = true;
	this->_inputData[2]->print_data();
	this->diff.print_data();
	Data<Dtype>::printConfig = false;
#endif
		// apply "inside" weights
		soooa_gpu_mul(
				count,
				this->_inputData[2]->device_data(),
				this->diff.device_data(),
				this->diff.mutable_device_data());	// d := w_in * (b0 - b1)

#if SMOOTHL1LOSSLAYER_LOG
	Data<Dtype>::printConfig = true;
	this->diff.print_data();
	Data<Dtype>::printConfig = false;
#endif

	}

	// smoothL1Forward
	const float sigma2 = SLPROP(SmoothL1Loss, sigma) * SLPROP(SmoothL1Loss, sigma);
	SmoothL1Forward<Dtype><<<SOOOA_GET_BLOCKS(count), SOOOA_CUDA_NUM_THREADS>>>(
	      count, this->diff.device_data(), this->errors.mutable_device_data(), sigma2);
	CUDA_POST_KERNEL_CHECK;

#if SMOOTHL1LOSSLAYER_LOG
	Data<Dtype>::printConfig = true;
	this->diff.print_data();
	this->errors.print_data();
	Data<Dtype>::printConfig = false;
#endif

	if (this->hasWeights) {

#if SMOOTHL1LOSSLAYER_LOG
	Data<Dtype>::printConfig = true;
	this->_inputData[3]->print_data();
	this->errors.print_data();
	Data<Dtype>::printConfig = false;
#endif

		// apply "outside" weights
		soooa_gpu_mul(
				count,
				this->_inputData[3]->device_data(),
				this->errors.device_data(),
				this->errors.mutable_device_data());	// d := w_out * SmoothL1(w_in * (b0 - b1))

#if SMOOTHL1LOSSLAYER_LOG
	Data<Dtype>::printConfig = true;
	this->errors.print_data();
	Data<Dtype>::printConfig = false;
#endif
	}

	const uint32_t firstAxis = SLPROP(SmoothL1Loss, firstAxis);
	const float lossWeight = SLPROP(Loss, lossWeight);
	Dtype loss;
	soooa_gpu_dot(count, this->ones.device_data(), this->errors.device_data(), &loss);
	this->_outputData[0]->mutable_host_data()[0] = loss * Dtype(lossWeight) /
			this->_inputData[0]->getShape(firstAxis);
	//this->_outputData[0]->mutable_host_data()[0] = loss * Dtype(this->lossWeight);
	//cout << "smoothl1loss: " << this->_outputData[0]->host_data()[0] << endl;
}


template <typename Dtype>
__global__ void SmoothL1Backward(const uint32_t n, const Dtype* in, Dtype* out,
    Dtype sigma2) {
  // f'(x) = sigma2 * sigma2 * x         if |x| < 1 / sigma2 / sigma2
  //       = sign(x)                   otherwise
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = in[index];
    Dtype abs_val = abs(val);
    if (abs_val < 1.0 / sigma2) {
      out[index] = sigma2 * val;
    } else {
      out[index] = (Dtype(0) < val) - (val < Dtype(0));
    }
  }
}


template <typename Dtype>
void SmoothL1LossLayer<Dtype>::backpropagation() {
	// after forwards, diff holds w_in * (b0 - b1)

	/*
	if (this->name == "rpn_loss_bbox") {
		Data<Dtype>::printConfig = true;
		this->diff.print_data({}, false);
		Data<Dtype>::printConfig = false;
	}
	*/

	const float sigma2 = SLPROP(SmoothL1Loss, sigma) * SLPROP(SmoothL1Loss, sigma);
	const uint32_t count = this->diff.getCount();
	SmoothL1Backward<Dtype><<<SOOOA_GET_BLOCKS(count), SOOOA_CUDA_NUM_THREADS>>>(
			count, this->diff.device_data(), this->diff.mutable_device_data(), sigma2);
	CUDA_POST_KERNEL_CHECK;

	/*
	if (this->name == "rpn_loss_bbox") {
		Data<Dtype>::printConfig = true;
		this->diff.print_data({}, false);
		Data<Dtype>::printConfig = false;
	}
	*/

	const vector<bool> propDown = SLPROP_BASE(propDown);
	const uint32_t firstAxis = SLPROP(SmoothL1Loss, firstAxis);
	for (uint32_t i = 0; i < 2; i++) {
		if (propDown[i]) {
			const Dtype sign = (i == 0) ? 1 : -1;
			// XXX: caffe, top[0]->cpu_diff()[0]에 대해서 set하는 부분을 찾을 수 없고
			// 현재 특수한 값이 들어 있는 것이 아닌 1의 값이 들어있어 상수 1.0f으로 대체
			//const Dtype alpha = sign * this->_outputData[0]->host_grad()[0] /
			//		this->_inputData[i]->batches();

			const Dtype alpha = sign * Dtype(1) /
					this->_inputData[i]->getShape(firstAxis);
			soooa_gpu_axpby(
					count,
					alpha,
					this->diff.device_data(),
					Dtype(0),
					this->_inputData[i]->mutable_device_grad());
			if (this->hasWeights) {
				// Scale by "inside" weight
				soooa_gpu_mul(
						count,
						this->_inputData[2]->device_data(),
						this->_inputData[i]->device_grad(),
						this->_inputData[i]->mutable_device_grad());
				// Scale by "outside" weight
				soooa_gpu_mul(
						count,
						this->_inputData[3]->device_data(),
						this->_inputData[i]->device_grad(),
						this->_inputData[i]->mutable_device_grad());
			}
		}
	}

	/*
	if (this->name == "rpn_loss_bbox") {
		Data<Dtype>::printConfig = true;
		this->_inputData[i]->print_grad({}, false);
		Data<Dtype>::printConfig = false;
	}
	*/
}

template <typename Dtype>
Dtype SmoothL1LossLayer<Dtype>::cost() {
	return this->_outputData[0]->host_data()[0];
}




template <typename Dtype>
void SmoothL1LossLayer<Dtype>::initialize() {

}




/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* SmoothL1LossLayer<Dtype>::initLayer() {
    SmoothL1LossLayer* layer = new SmoothL1LossLayer<Dtype>(SLPROP_BASE(name));
    return (void*)layer;
}

template<typename Dtype>
void SmoothL1LossLayer<Dtype>::destroyLayer(void* instancePtr) {
    SmoothL1LossLayer<Dtype>* layer = (SmoothL1LossLayer<Dtype>*)instancePtr;
    delete layer;
}

template<typename Dtype>
void SmoothL1LossLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
    SASSERT0(index == 0);

    SmoothL1LossLayer<Dtype>* layer = (SmoothL1LossLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == 0);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == 0);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool SmoothL1LossLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    SmoothL1LossLayer<Dtype>* layer = (SmoothL1LossLayer<Dtype>*)instancePtr;
    //layer->reshape();
    return true;
}

template<typename Dtype>
void SmoothL1LossLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
    cout << "SmoothL1LossLayer.. forward(). miniBatchIndex : " << miniBatchIdx << endl;
}

template<typename Dtype>
void SmoothL1LossLayer<Dtype>::backwardTensor(void* instancePtr) {
    cout << "SmoothL1LossLayer.. backward()" << endl;
}

template<typename Dtype>
void SmoothL1LossLayer<Dtype>::learnTensor(void* instancePtr) {
    cout << "SmoothL1LossLayer.. learn()" << endl;
}



template class SmoothL1LossLayer<float>;





