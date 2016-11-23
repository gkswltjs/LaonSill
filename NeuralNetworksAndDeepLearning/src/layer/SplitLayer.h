/*
 * SplitLayer.h
 *
 *  Created on: Nov 8, 2016
 *      Author: jkim
 */

#ifndef SPLITLAYER_H_
#define SPLITLAYER_H_



#include "HiddenLayer.h"

template <typename Dtype>
class SplitLayer : public HiddenLayer<Dtype> {
public:
	class Builder : public HiddenLayer<Dtype>::Builder {
	public:
		Builder() {
			this->type = Layer<Dtype>::Split;
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
		Layer<Dtype>* build() {
			return new SplitLayer(this);
		}
		virtual void save(std::ofstream& ofs) {
			Layer<Dtype>::Builder::save(ofs);
		}
		virtual void load(std::ifstream& ifs) {
			Layer<Dtype>::Builder::load(ifs);
		}
	};

	SplitLayer(const std::string& name);
	SplitLayer(Builder* builder);
	virtual ~SplitLayer();

	virtual void reshape();
	virtual void feedforward();

private:
	void initialize();


	virtual void backpropagation();

};

#endif /* SPLITLAYER_H_ */
