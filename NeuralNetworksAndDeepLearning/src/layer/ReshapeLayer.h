/*
 * ReshapeLayer.h
 *
 *  Created on: Nov 23, 2016
 *      Author: jkim
 */

#ifndef RESHAPELAYER_H_
#define RESHAPELAYER_H_


#if 1
#include <vector>

#include "common.h"
#include "HiddenLayer.h"


template <typename Dtype>
class ReshapeLayer : public HiddenLayer<Dtype> {
public:
	class Builder : public HiddenLayer<Dtype>::Builder {
	public:
		std::vector<int> _shape;
		int _axis;
		int _numAxes;

		Builder() {
			this->type = Layer<Dtype>::Reshape;
			this->_axis = 0;
			this->_numAxes = -1;
		}
		Builder* shape(const std::vector<int>& shape) {
			this->_shape = shape;
			return this;
		}
		Builder* axis(const int axis) {
			this->_axis = axis;
			return this;
		}
		Builder* numAxes(const int numAxes) {
			this->_numAxes = numAxes;
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
		virtual Builder* propDown(const std::vector<bool>& propDown) {
			HiddenLayer<Dtype>::Builder::propDown(propDown);
			return this;
		}
		Layer<Dtype>* build() {
			return new ReshapeLayer(this);
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
	std::vector<int> shape;
	int axis;
	int numAxes;
	std::vector<uint32_t> copyAxes;
	int inferredAxis;
	uint32_t constantCount;
};

#endif

#endif /* RESHAPELAYER_H_ */
