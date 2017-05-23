/*
 * SplitLayer.h
 *
 *  Created on: Nov 8, 2016
 *      Author: jkim
 */

#ifndef SPLITLAYER_H_
#define SPLITLAYER_H_



#include "Layer.h"

template <typename Dtype>
class SplitLayer : public Layer<Dtype> {
public:
	class Builder : public Layer<Dtype>::Builder {
	public:
		Builder() {
			this->type = Layer<Dtype>::Split;
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
		Layer<Dtype>* build() {
			return new SplitLayer(this);
		}
	};

	SplitLayer(const std::string& name);
	SplitLayer(Builder* builder);
	virtual ~SplitLayer();

	virtual void reshape();
	virtual void feedforward();

private:
	void initialize();


	virtual void backpropagation();

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

#endif /* SPLITLAYER_H_ */
