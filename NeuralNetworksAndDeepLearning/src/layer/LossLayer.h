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

	LossLayer() : Layer<Dtype>() {}
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
	NormalizationMode normalization;

};

#endif /* LOSSLAYER_H_ */
