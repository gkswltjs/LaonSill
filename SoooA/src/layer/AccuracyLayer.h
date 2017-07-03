/*
 * AccuracyLayer.h
 *
 *  Created on: Apr 25, 2017
 *      Author: jkim
 */

#ifndef ACCURACYLAYER_H_
#define ACCURACYLAYER_H_

#include "common.h"
#include "BaseLayer.h"

template <typename Dtype>
class AccuracyLayer : public Layer<Dtype> {
public:
	/*
	class Builder : public Layer<Dtype>::Builder {
	public:
		uint32_t _topK;
		int _axis;
		int _ignoreLabel;

		Builder() {
			this->type = Layer<Dtype>::Accuracy;
			this->_topK = 1;
			this->_axis = 1;
			this->_ignoreLabel = -1;
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
		virtual Builder* topK(const uint32_t topK) {
			this->_topK = topK;
			return this;
		}
		virtual Builder* axis(const int axis) {
			this->_axis = axis;
			return this;
		}
		virtual Builder* ignoreLabel(const int ignoreLabel) {
			this->_ignoreLabel = ignoreLabel;
			return this;
		}
		Layer<Dtype>* build() {
			return new AccuracyLayer(this);
		}
	};
	*/


	AccuracyLayer();
	virtual ~AccuracyLayer();

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();

	Dtype getAccuracy();


private:
	//uint32_t topK;
	//int labelAxis;

	bool hasIgnoreLabel;
	//int ignoreLabel;

	int outerNum;
	int innerNum;


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

#endif /* ACCURACYLAYER_H_ */
