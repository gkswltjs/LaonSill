/*
 * DummyInputLayer.h
 *
 *  Created on: Jan 21, 2017
 *      Author: jkim
 */

#ifndef DUMMYINPUTLAYER_H_
#define DUMMYINPUTLAYER_H_

#include "InputLayer.h"

template <typename Dtype>
class DummyInputLayer
: public InputLayer<Dtype> {

public:
	class Builder : public InputLayer<Dtype>::Builder {
	public:
		Builder() {
			this->type = Layer<Dtype>::Input;
		}
		virtual Builder* name(const std::string name) {
			InputLayer<Dtype>::Builder::name(name);
			return this;
		}
		virtual Builder* id(uint32_t id) {
			InputLayer<Dtype>::Builder::id(id);
			return this;
		}
		virtual Builder* inputs(const std::vector<std::string>& inputs) {
			InputLayer<Dtype>::Builder::inputs(inputs);
			return this;
		}
		virtual Builder* outputs(const std::vector<std::string>& outputs) {
			InputLayer<Dtype>::Builder::outputs(outputs);
			return this;
		}
		virtual Builder* propDown(const std::vector<bool>& propDown) {
			InputLayer<Dtype>::Builder::propDown(propDown);
			return this;
		}
		Layer<Dtype>* build() {
			return new DummyInputLayer(this);
		}
	};

	DummyInputLayer(Builder* builder);
	virtual ~DummyInputLayer();

    int getNumTrainData();
    int getNumTestData();
    void shuffleTrainDataSet();

	virtual void feedforward();
	virtual void reshape();

private:
	void initialize();


};

#endif /* DUMMYINPUTLAYER_H_ */
