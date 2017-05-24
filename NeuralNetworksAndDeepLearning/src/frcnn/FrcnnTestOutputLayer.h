/*
 * FrcnnTestOutputLayer.h
 *
 *  Created on: Dec 16, 2016
 *      Author: jkim
 */

#ifndef FRCNNTESTOUTPUTLAYER_H_
#define FRCNNTESTOUTPUTLAYER_H_

#include "Layer.h"
#include "frcnn_common.h"

template <typename Dtype>
class FrcnnTestOutputLayer : public Layer<Dtype> {
public:
	class Builder : public Layer<Dtype>::Builder {
	public:
		uint32_t _maxPerImage;
		float _thresh;
		bool _vis;

		Builder() {
			this->type = Layer<Dtype>::FrcnnTestOutput;
			this->_maxPerImage = 100;
			this->_thresh = 0.05f;
			this->_vis = false;
		}
		virtual Builder* maxPerImage(const uint32_t maxPerImage) {
			this->_maxPerImage = maxPerImage;
			return this;
		}
		virtual Builder* thresh(const float thresh) {
			this->_thresh = thresh;
			return this;
		}
		virtual Builder* vis(const bool vis) {
			this->_vis = vis;
			return this;
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
			return new FrcnnTestOutputLayer(this);
		}
	};

	FrcnnTestOutputLayer(const std::string& name);
	FrcnnTestOutputLayer(Builder* builder);
	virtual ~FrcnnTestOutputLayer();

	virtual void reshape();
	virtual void feedforward();

private:
	void initialize();


public:
	uint32_t maxPerImage;
	float thresh;
	bool vis;

	std::vector<cv::Scalar> boxColors;


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

#endif /* FRCNNTESTOUTPUTLAYER_H_ */
