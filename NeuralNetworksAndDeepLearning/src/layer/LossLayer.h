/*
 * LossLayer.h
 *
 *  Created on: Nov 24, 2016
 *      Author: jkim
 */

#ifndef LOSSLAYER_H_
#define LOSSLAYER_H_

#include <vector>

#include "common.h"
#include "Layer.h"

template <typename Dtype>
class LossLayer : public Layer<Dtype> {
public:
	enum NormalizationMode {
		Full = 0,
		Valid,
		BatchSize,
		NoNormalization
	};

	class Builder : public Layer<Dtype>::Builder {
	public:
		float _lossWeight;

		bool _hasIgnoreLabel;
		int _ignoreLabel;

		bool _hasNormalize;
		bool _normalize;

		bool _hasNormalization;
		NormalizationMode _normalization;

		Builder() {
			this->_lossWeight = 1.0f;
			this->_hasIgnoreLabel = false;
			this->_ignoreLabel = -1;
			this->_hasNormalize = false;
			this->_normalize = false;
			this->_hasNormalization = false;
			this->_normalization = NormalizationMode::Valid;
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
		virtual Builder* lossWeight(const float lossWeight) {
			this->_lossWeight = lossWeight;
			return this;
		}
		virtual Builder* ignoreLabel(const int ignoreLabel) {
			this->_hasIgnoreLabel = true;
			this->_ignoreLabel = ignoreLabel;
			return this;
		}
		virtual Builder* normalize(const bool normalize) {
			this->_hasNormalize = true;
			this->_normalize = normalize;
			return this;
		}
		virtual Builder* normalization(const NormalizationMode normalization) {
			this->_hasNormalization = true;
			this->_normalization = normalization;
			return this;
		}
		virtual Builder* propDown(const std::vector<bool>& propDown) {
			Layer<Dtype>::Builder::propDown(propDown);
			return this;
		}
		Layer<Dtype>* build() = 0;

	};


	LossLayer()
		: Layer<Dtype>() {}
	LossLayer(const std::string& name)
		: Layer<Dtype>(name) {}
	LossLayer(Builder* builder)
		: Layer<Dtype>(builder) {

		/*
		if (builder->_propDown.size() > 0)
			this->propDown = builder->_propDown;
		else {
			this->propDown.resize(this->_inputData.size());
			for (uint32_t i = 0; i < this->_inputData.size(); i++)
				this->propDown[i] = true;
		}
		*/

		//this->propDown = builder->_propDown;
		this->lossWeight = builder->_lossWeight;

		this->hasIgnoreLabel = builder->_hasIgnoreLabel;
		this->ignoreLabel = builder->_ignoreLabel;

		this->hasNormalize = builder->_hasNormalize;
		this->normalize = builder->_normalize;

		this->hasNormalization = builder->_hasNormalization;
		this->normalization = builder->_normalization;
	}
	virtual ~LossLayer() {}

	virtual void reshape() {
		Layer<Dtype>::reshape();
	}
	virtual void feedforward() {
		Layer<Dtype>::feedforward();
	}
	virtual void backpropagation() {
		Layer<Dtype>::backpropagation();
	}
	virtual Dtype cost() = 0;

protected:

	float lossWeight;

	bool hasIgnoreLabel;
	int ignoreLabel;

	bool hasNormalize;
	bool normalize;

	bool hasNormalization;
	NormalizationMode normalization;

};

#endif /* LOSSLAYER_H_ */
