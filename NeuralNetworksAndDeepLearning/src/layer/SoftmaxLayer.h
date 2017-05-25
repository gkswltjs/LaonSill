/*
 * SoftmaxLayer.h
 *
 *  Created on: Nov 29, 2016
 *      Author: jkim
 */

#ifndef SOFTMAXLAYER_H_
#define SOFTMAXLAYER_H_

#if 1

#include "common.h"
#include "Layer.h"
//#include "Activation.h"
#include "Cuda.h"

template <typename Dtype>
class SoftmaxLayer : public Layer<Dtype> {
public:
	class Builder : public Layer<Dtype>::Builder {
	public:
		uint32_t _softmaxAxis;

		Builder() {
			this->type = Layer<Dtype>::Softmax;
			this->_softmaxAxis = 2;
		}
		virtual Builder* name(const std::string name) {
			Layer<Dtype>::Builder::name(name);
			return this;
		}
		virtual Builder* id(uint32_t id) {
			Layer<Dtype>::Builder::id(id);
			return this;
		}
		virtual Builder* inputs(const std::vector<std::string>& inputs) {
			Layer<Dtype>::Builder::inputs(inputs);
			return this;
		}
		virtual Builder* outputs(const std::vector<std::string>& outputs) {
			Layer<Dtype>::Builder::outputs(outputs);
			return this;
		}
		virtual Builder* propDown(const std::vector<bool>& propDown) {
			Layer<Dtype>::Builder::propDown(propDown);
			return this;
		}
		virtual Builder* softmaxAxis(const uint32_t softmaxAxis) {
			this->_softmaxAxis = softmaxAxis;
			return this;
		}
		Layer<Dtype>* build() {
			return new SoftmaxLayer(this);
		}
	};

	SoftmaxLayer(const std::string& name);
	SoftmaxLayer(Builder* builder);

	virtual ~SoftmaxLayer();

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();

private:
	void initialize();

	uint32_t softmaxAxis;
	uint32_t outerNum;
	uint32_t innerNum;

	//Activation<Dtype> *activation_fn;

	// used to carry out sum using BLAS
	Data<Dtype> sumMultiplier;
	// intermediate data to hold temporary results.
	Data<Dtype> scale;

	cudnnTensorDescriptor_t inputTensorDesc;
	cudnnTensorDescriptor_t outputTensorDesc;


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

#endif /* SOFTMAXLAYER_H_ */
