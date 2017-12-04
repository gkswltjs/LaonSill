/*
 * PriorBoxLayer.h
 *
 *  Created on: Apr 22, 2017
 *      Author: jkim
 */

#ifndef PRIORBOXLAYER_H_
#define PRIORBOXLAYER_H_

#include "common.h"
#include "BaseLayer.h"

/*
 * @brief Generate the prior boxes of designated sizes and aspect ratios across
 *        all dimensions
 */
template <typename Dtype>
class PriorBoxLayer : public Layer<Dtype> {
public:
	/*
	class Builder : public Layer<Dtype>::Builder {
	public:
		std::vector<Dtype> _minSizes;
		std::vector<Dtype> _maxSizes;
		std::vector<Dtype> _aspectRatios;
		bool _flip;
		bool _clip;
		std::vector<Dtype> _variances;

		int _imgSize;
		int _imgH;
		int _imgW;

		Dtype _step;
		Dtype _stepH;
		Dtype _stepW;

		Dtype _offset;



		Builder() {
			this->type = Layer<Dtype>::PriorBox;
			this->_flip = true;
			this->_clip = false;
			this->_imgSize = -1;
			this->_imgH = -1;
			this->_imgW = -1;
			this->_step = Dtype(-1);
			this->_stepH = Dtype(-1);
			this->_stepW = Dtype(-1);
			this->_offset = Dtype(0.5);
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
		virtual Builder* minSizes(const std::vector<Dtype> minSizes) {
			this->_minSizes = minSizes;
			return this;
		}
		virtual Builder* maxSizes(const std::vector<Dtype> maxSizes) {
			this->_maxSizes = maxSizes;
			return this;
		}
		virtual Builder* aspectRatios(const std::vector<Dtype> aspectRatios) {
			this->_aspectRatios = aspectRatios;
			return this;
		}
		virtual Builder* flip(const bool flip) {
			this->_flip = flip;
			return this;
		}
		virtual Builder* clip(const bool clip) {
			this->_clip = clip;
			return this;
		}
		virtual Builder* variances(const std::vector<Dtype>& variances) {
			this->_variances = variances;
			return this;
		}
		virtual Builder* imgSize(const uint32_t imgSize) {
			this->_imgSize = imgSize;
			return this;
		}
		virtual Builder* imgH(const uint32_t imgH) {
			this->_imgH = imgH;
			return this;
		}
		virtual Builder* imgW(const uint32_t imgW) {
			this->_imgW = imgW;
			return this;
		}
		virtual Builder* step(const Dtype step) {
			this->_step = step;
			return this;
		}
		virtual Builder* stepH(const Dtype stepH) {
			this->_stepH = stepH;
			return this;
		}
		virtual Builder* stepW(const Dtype stepW) {
			this->_stepW = stepW;
			return this;
		}
		virtual Builder* offset(const Dtype offset) {
			this->_offset = offset;
			return this;
		}
		Layer<Dtype>* build() {
			return new PriorBoxLayer(this);
		}
	};
	*/



	PriorBoxLayer();
	virtual ~PriorBoxLayer();

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();

private:



private:
	/*
	std::vector<Dtype> minSizes;
	std::vector<Dtype> maxSizes;
	std::vector<Dtype> aspectRatios;
	bool flip;
	bool clip;
	std::vector<Dtype> variances;
	uint32_t imgW;
	uint32_t imgH;
	Dtype stepW;
	Dtype stepH;
	Dtype offset;
	*/

	int numPriors;





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

#endif /* PRIORBOXLAYER_H_ */
