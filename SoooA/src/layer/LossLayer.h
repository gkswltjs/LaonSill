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
#include "BaseLayer.h"
#include "EnumDef.h"
#include "PropMgmt.h"

template <typename Dtype>
class LossLayer : public Layer<Dtype> {
public:
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
		virtual Dtype getNormalizer(const NormalizationMode normalizationMode,
				const int outerNum, const int innerNum, const int validCount) {
		Dtype normalizer;
		switch (normalizationMode) {
		case NormalizationMode::Full:
			normalizer = Dtype(outerNum * innerNum);
			break;
		case NormalizationMode::Valid:
			if (validCount == -1) {
				normalizer = Dtype(outerNum * innerNum);
			} else {
				normalizer = Dtype(validCount);
			}
			break;
		case NormalizationMode::BatchSize:
			normalizer = Dtype(outerNum);
			break;
		case NormalizationMode::NoNormalization:
			normalizer = Dtype(1);
			break;
		default:
			SASSERT(false, "Unknown normlization mode.");
		}
		// Some useres will have no labels for some examples in order to 'turn off' a
		// particular loss in a multi-task setup. The max prevents NaNs in that case.
		return std::max(Dtype(1.0), normalizer);
	}
	
};

#endif /* LOSSLAYER_H_ */
