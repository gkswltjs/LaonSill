/*
 * RoIPoolingLayer.h
 *
 *  Created on: Dec 1, 2016
 *      Author: jkim
 */

#ifndef ROIPOOLINGLAYER_H_
#define ROIPOOLINGLAYER_H_

#include <vector>

#include "common.h"
#include "Layer.h"

template <typename Dtype>
class RoIPoolingLayer : public Layer<Dtype> {
public:
	class Builder : public Layer<Dtype>::Builder {
	public:
		uint32_t _pooledW;
		uint32_t _pooledH;
		float _spatialScale;

		Builder() {
			this->type = Layer<Dtype>::RoIPooling;
			this->_pooledW = 6;
			this->_pooledH = 6;
			this->_spatialScale = 0.0625;	// 1/16
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
		Builder* pooledW(const uint32_t pooledW) {
			this->_pooledW = pooledW;
			return this;
		}
		Builder* pooledH(const uint32_t pooledH) {
			this->_pooledH = pooledH;
			return this;
		}
		Builder* spatialScale(const float spatialScale) {
			this->_spatialScale = spatialScale;
			return this;
		}
		Layer<Dtype>* build() {
			return new RoIPoolingLayer(this);
		}
	};

	RoIPoolingLayer(const std::string& name);
	RoIPoolingLayer(Builder* builder);
	virtual ~RoIPoolingLayer();

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();

private:
	void initialize();


private:
	uint32_t pooledW;
	uint32_t pooledH;
	float spatialScale;

	uint32_t channels;
	uint32_t height;
	uint32_t width;
	Data<int> maxIdx;


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

#endif /* ROIPOOLINGLAYER_H_ */
