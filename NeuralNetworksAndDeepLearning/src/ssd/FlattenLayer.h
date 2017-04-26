/*
 * FlattenLayer.h
 *
 *  Created on: Apr 22, 2017
 *      Author: jkim
 */

#ifndef FLATTENLAYER_H_
#define FLATTENLAYER_H_

#include "common.h"
#include "Layer.h"

/*
 * @breif Reshapes the input data into flat vectors
 */
template <typename Dtype>
class FlattenLayer : public Layer<Dtype> {
public:
	class Builder : public Layer<Dtype>::Builder {
	public:
		uint32_t _axis;

		Builder() {
			this->type = Layer<Dtype>::Flatten;
			this->_axis = 1;
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
		virtual Builder* axis(const uint32_t axis) {
			this->_axis = axis;
			return this;
		}
		Layer<Dtype>* build() {
			return new FlattenLayer(this);
		}
	};

	FlattenLayer(Builder* builder);
	virtual ~FlattenLayer();

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();

private:
	void initialize();

private:
	uint32_t axis;
};

#endif /* FLATTENLAYER_H_ */