/*
 * SoftmaxWithLoss.h
 *
 *  Created on: Nov 23, 2016
 *      Author: jkim
 */

#ifndef SOFTMAXWITHLOSSLAYER_H_
#define SOFTMAXWITHLOSSLAYER_H_

#if 1
#include "common.h"
#include "LossLayer.h"
#include "SoftmaxLayer.h"
#include "Activation.h"
#include "Cost.h"
#include "Cuda.h"

template <typename Dtype>
class SoftmaxWithLossLayer : public LossLayer<Dtype> {
public:
	class Builder : public LossLayer<Dtype>::Builder {
	public:
		uint32_t _softmaxAxis;

		Builder() {
			this->type = Layer<Dtype>::SoftmaxWithLoss;
			this->_softmaxAxis = 2;
		}
		virtual Builder* name(const std::string name) {
			LossLayer<Dtype>::Builder::name(name);
			return this;
		}
		virtual Builder* id(uint32_t id) {
			LossLayer<Dtype>::Builder::id(id);
			return this;
		}
		virtual Builder* inputs(const std::vector<std::string>& inputs) {
			LossLayer<Dtype>::Builder::inputs(inputs);
			return this;
		}
		virtual Builder* outputs(const std::vector<std::string>& outputs) {
			LossLayer<Dtype>::Builder::outputs(outputs);
			return this;
		}
		virtual Builder* propDown(const std::vector<bool>& propDown) {
			LossLayer<Dtype>::Builder::propDown(propDown);
			return this;
		}
		virtual Builder* lossWeight(const float lossWeight) {
			LossLayer<Dtype>::Builder::lossWeight(lossWeight);
			return this;
		}
		virtual Builder* ignoreLabel(const int ignoreLabel) {
			LossLayer<Dtype>::Builder::ignoreLabel(ignoreLabel);
			return this;
		}
		virtual Builder* normalize(const bool normalize) {
			LossLayer<Dtype>::Builder::normalize(normalize);
			return this;
		}

		/*
		virtual Builder* normalizationMode(
            const typename LossLayer<Dtype>::NormalizationMode normalizationMode) {
			LossLayer<Dtype>::Builder::normalizationMode(normalizationMode);
			return this;
		}
		*/

		virtual Builder* softmaxAxis(const uint32_t softmaxAxis) {
			this->_softmaxAxis = softmaxAxis;
			return this;
		}
		Layer<Dtype>* build() {
			if (this->_propDown.size() != this->_inputs.size()) {
				this->_propDown.resize(this->_inputs.size());

				for (uint32_t i = 0; i < this->_inputs.size(); i++) {
					if (i == 0)
						this->_propDown[0] = true;
					else
						this->_propDown[i] = false;
				}
			}
			return new SoftmaxWithLossLayer(this);
		}
	};

	SoftmaxWithLossLayer(Builder* builder);
    SoftmaxWithLossLayer(const std::string& name);
	virtual ~SoftmaxWithLossLayer();

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();
	virtual Dtype cost();

private:
	void initialize();
	Dtype getNormalizer(int validCount);

public:
	Data<Dtype>* prob;

private:
	uint32_t softmaxAxis;
	uint32_t outerNum;
	uint32_t innerNum;

	SoftmaxLayer<Dtype>* softmaxLayer;

	Activation<Dtype> *activation_fn;
	Cost<Dtype>* cost_fn;

	cudnnTensorDescriptor_t inputTensorDesc;
	cudnnTensorDescriptor_t probTensorDesc;

public:
    /****************************************************************************
     * layer callback functions 
     ****************************************************************************/
    static void* initLayer();
    static void destroyLayer(void* instancePtr);
    static void setInOutTensor(void* instancePtr, void* tensorPtr, bool isInput, int index);
    static bool allocLayerTensors(void* instancePtr);
    static void forwardTensor(void* instancePtr, int miniBatchIndex);
    static void backwardTensor(void* instancePtr);
    static void learnTensor(void* instancePtr);
};
#endif

#endif /* SOFTMAXWITHLOSSLAYER_H_ */
