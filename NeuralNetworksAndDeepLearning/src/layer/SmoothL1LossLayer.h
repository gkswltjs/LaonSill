/*
 * SmoothL1LossLayer.h
 *
 *  Created on: Nov 23, 2016
 *      Author: jkim
 */

#ifndef SMOOTHL1LOSSLAYER_H_
#define SMOOTHL1LOSSLAYER_H_


#if 0
#include "common.h"
#include "HiddenLayer.h"

template <typename Dtype>
class SmoothL1LossLayer : public HiddenLayer<Dtype> {
public:
	class Builder : public HiddenLayer<Dtype>::Builder {
	public:
		float _sigma;

		Builder() {
			this->type = Layer<Dtype>::Pool;
		}
		Builder* sigma(float sigma) {
			this->_sigma = sigma;
			return this;
		}
		virtual Builder* name(const std::string name) {
			HiddenLayer<Dtype>::Builder::name(name);
			return this;
		}
		virtual Builder* id(uint32_t id) {
			HiddenLayer<Dtype>::Builder::id(id);
			return this;
		}
		virtual Builder* inputs(const std::vector<std::string>& inputs) {
			HiddenLayer<Dtype>::Builder::inputs(inputs);
			return this;
		}
		virtual Builder* outputs(const std::vector<std::string>& outputs) {
			HiddenLayer<Dtype>::Builder::outputs(outputs);
			return this;
		}
		Layer<Dtype>* build() {
			return new SmoothL1LossLayer(this);
		}
		virtual void save(std::ofstream& ofs) {
			HiddenLayer<Dtype>::Builder::save(ofs);
			ofs.write((char*)&_poolDim, sizeof(pool_dim));
			ofs.write((char*)&_poolingType, sizeof(typename Pooling<Dtype>::Type));
		}
		virtual void load(std::ifstream& ifs) {
			HiddenLayer<Dtype>::Builder::load(ifs);
			ifs.read((char*)&_poolDim, sizeof(pool_dim));
			ifs.read((char*)&_poolingType, sizeof(typename Pooling<Dtype>::Type));
		}
	};

	SmoothL1LossLayer();
	SmoothL1LossLayer(Builder* builder);
	virtual ~SmoothL1LossLayer();

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();

private:
	void initialize();

private:
	float sigma;
};

#endif

#endif /* SMOOTHL1LOSSLAYER_H_ */
