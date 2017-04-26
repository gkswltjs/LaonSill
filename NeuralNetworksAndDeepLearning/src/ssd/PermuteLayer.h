/*
 * PermuteLayer.h
 *
 *  Created on: Apr 22, 2017
 *      Author: jkim
 */

#ifndef PERMUTELAYER_H_
#define PERMUTELAYER_H_

#include "common.h"
#include "Layer.h"

/*
 * @brief Permute the input blob by changing the memory order of the data
 */
template <typename Dtype>
class PermuteLayer : public Layer<Dtype> {
public:
	class Builder : public Layer<Dtype>::Builder {
	public:
		std::vector<uint32_t> _orders;
		Builder() {
			this->type = Layer<Dtype>::Permute;
			this->_orders = {0, 1, 2, 3};
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
		virtual Builder* orders(const std::vector<uint32_t>& orders) {
			this->_orders = orders;
			return this;
		}
		Layer<Dtype>* build() {
			return new PermuteLayer(this);
		}
	};


	PermuteLayer(Builder* builder);
	virtual ~PermuteLayer();

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();

private:
	void initialize();


private:
	std::vector<uint32_t> orders;

	uint32_t numAxes;
	bool needPermute;

	Data<int> permuteOrder_;
	Data<int> oldSteps_;
	Data<int> newSteps_;
};

#endif /* PERMUTELAYER_H_ */
