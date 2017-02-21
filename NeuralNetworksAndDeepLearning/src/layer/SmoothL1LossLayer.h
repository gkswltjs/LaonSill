/*
 * SmoothL1LossLayer.h
 *
 *  Created on: Nov 23, 2016
 *      Author: jkim
 */

#ifndef SMOOTHL1LOSSLAYER_H_
#define SMOOTHL1LOSSLAYER_H_


#include "common.h"
#include "LossLayer.h"

template <typename Dtype>
class SmoothL1LossLayer : public LossLayer<Dtype> {
public:
	class Builder : public LossLayer<Dtype>::Builder {
	public:
		float _sigma;
		uint32_t _firstAxis;

		Builder() {
			this->type = Layer<Dtype>::SmoothL1Loss;
			this->_sigma = 1.0f;
			this->_firstAxis = 0;
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
		Builder* sigma(float sigma) {
			this->_sigma = sigma;
			return this;
		}
		Builder* firstAxis(uint32_t firstAxis) {
			this->_firstAxis = firstAxis;
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
			return new SmoothL1LossLayer(this);
		}
	};

	SmoothL1LossLayer(Builder* builder);
	virtual ~SmoothL1LossLayer();

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();
	virtual Dtype cost();

	virtual inline int exactNumBottomBlobs() const { return -1; }
	virtual inline int minBottomBlobs() const { return 2; }
	virtual inline int maxBottomBlobs() const { return 4; }

private:
	void initialize();

private:
	Data<Dtype>* diff;
	Data<Dtype>* errors;
	Data<Dtype>* ones;
	bool hasWeights;
	float sigma2;
	uint32_t firstAxis;
};

#endif /* SMOOTHL1LOSSLAYER_H_ */
