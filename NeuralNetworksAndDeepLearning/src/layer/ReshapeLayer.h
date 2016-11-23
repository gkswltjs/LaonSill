/*
 * ReshapeLayer.h
 *
 *  Created on: Nov 23, 2016
 *      Author: jkim
 */

#ifndef RESHAPELAYER_H_
#define RESHAPELAYER_H_


#if 0
#include <vector>

#include "common.h"
#include "HiddenLayer.h"


template <typename Dtype>
class ReshapeLayer : public HiddenLayer<Dtype> {
public:
	class Builder : public HiddenLayer<Dtype>::Builder {
	public:
		std::vector<int> _shape;

		Builder() {
			this->type = Layer<Dtype>::Reshape;
		}
		Builder* shape(const std::vector<int>& shape) {
			this->_shape = shape;
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
			return new ReshapeLayer(this);
		}
		virtual void save(std::ofstream& ofs) {
			//HiddenLayer<Dtype>::Builder::save(ofs);
			//ofs.write((char*)&_poolDim, sizeof(pool_dim));
			//ofs.write((char*)&_poolingType, sizeof(typename Pooling<Dtype>::Type));
		}
		virtual void load(std::ifstream& ifs) {
			//HiddenLayer<Dtype>::Builder::load(ifs);
			//ifs.read((char*)&_poolDim, sizeof(pool_dim));
			//ifs.read((char*)&_poolingType, sizeof(typename Pooling<Dtype>::Type));
		}
	};

	ReshapeLayer();
	ReshapeLayer(Builder* builder);
	virtual ~ReshapeLayer();

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();

protected:
	void initialize();


private:
	std::vector<uint32_t> shape;
};

#endif

#endif /* RESHAPELAYER_H_ */
