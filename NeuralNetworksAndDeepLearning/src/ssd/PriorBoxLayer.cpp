/*
 * PriorBoxLayer.cpp
 *
 *  Created on: Apr 22, 2017
 *      Author: jkim
 */

#include "PriorBoxLayer.h"
#include "SysLog.h"

using namespace std;

template <typename Dtype>
PriorBoxLayer<Dtype>::PriorBoxLayer(Builder* builder)
: Layer<Dtype>(builder) {

	initialize(builder);
}

template <typename Dtype>
PriorBoxLayer<Dtype>::~PriorBoxLayer() {

}


template <typename Dtype>
void PriorBoxLayer<Dtype>::reshape() {

}

template <typename Dtype>
void PriorBoxLayer<Dtype>::feedforward() {
	reshape();
}

template <typename Dtype>
void PriorBoxLayer<Dtype>::backpropagation() {

}

template <typename Dtype>
void PriorBoxLayer<Dtype>::initialize(Builder* builder) {
	SASSERT(builder->_minSizes.size() > 0, "must provide minSizes.");
	for (int i = 0; i < builder->_minSizes.size(); i++) {
		this->minSizes.push_back(builder->_minSizes[i]);
		SASSERT(this->minSizes.back() > 0, "minSize must be positive.");
	}
	this->aspectRatios.clear();
	this->aspectRatios.push_back(Dtype(1));
	this->flip = builder->_flip;
	for (int i = 0; i < builder->_aspectRatios.size(); i++) {
		Dtype ar = builder->_aspectRatios[i];
		bool alreadyExsit = false;
		for (int j = 0; j < this->aspectRatios.size(); j++) {
			if (fabs(ar - this->aspectRatios[j]) < 1e-6) {
				alreadyExsit = true;
				break;
			}
		}
		if (!alreadyExsit) {
			this->aspectRatios.push_back(ar);
			if (this->flip) {
				this->aspectRatios.push_back(1 / ar);
			}
		}
	}
	this->numPriors = this->aspectRatios.size() * this->minSizes.size();
	if (builder->_maxSizes.size() > 0) {
		SASSERT0(builder->_minSizes.size() == builder->_maxSizes.size());
		for (int i = 0; i < builder->_maxSizes.size(); i++) {
			this->maxSizes.push_back(builder->_maxSizes[i]);
			SASSERT(this->maxSizes[i] > this->minSizes[i],
					"maxSize must b greater than minSize.");
			this->numPriors += 1;
		}
	}
	this->clip = builder->_clip;
	if (builder->_variances.size() > 1) {
		// Must and only provide 4 variance.
		SASSERT0(builder->_variances.size() == 4);
		for (int i = 0; i < builder->_variances.size(); i++) {
			SASSERT0(builder->_variances[i] > 0);
			this->variances.push_back(builder->_variances[i]);
		}
	} else if (builder->_variances.size() == 1) {
		SASSERT0(builder->_variances[0] > 0);
		this->variances.push_back(builder->_variances[0]);
	} else {
		// set tdefault to 0.1.
		this->variances.push_back(Dtype(0.1));
	}

	if (builder->_imgH >= 0 || builder->_imgW >= 0) {
		SASSERT(!(builder->_imgSize >= 0),
				"Either imgSize or imgH/imgW should be specified; not both.");
		this->imgH = builder->_imgH;
		SASSERT(this->imgH > 0, "imgH should be larger than 0.");
		this->imgW = builder->_imgW;
		SASSERT(this->imgW > 0, "imgW should be larger than 0.");
	} else if (builder->_imgSize >= 0) {
		SASSERT(builder->_imgSize > 0, "imgSize should be larger than 0.");
		this->imgH = builder->_imgSize;
		this->imgW = builder->_imgSize;
	} else {
		this->imgH = 0;
		this->imgW = 0;
	}

	if (builder->_stepH >= Dtype(0) || builder->_stepW >= Dtype(0)) {
		SASSERT(!(builder->_step >= 0),
				"Either step or stepH/stepW should be specified; not both.");
		this->stepH = builder->_stepH;
		SASSERT(this->stepH > Dtype(0), "stepH should be larger than 0.");
		this->stepW = builder->_stepW;
		SASSERT(this->stepW > Dtype(0), "stepW should be larger than 0.");
	} else if (builder->_step >= Dtype(0)) {
		SASSERT(builder->_step > Dtype(0), "step should be larger than 0.");
		this->stepH = builder->_step;
		this->stepW = builder->_step;
	} else {
		this->stepH = Dtype(0);
		this->stepW = Dtype(0);
	}

	this->offset = builder->_offset;
}

template class PriorBoxLayer<float>;





























































