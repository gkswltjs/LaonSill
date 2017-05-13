/*
 * PriorBoxLayer.cpp
 *
 *  Created on: Apr 22, 2017
 *      Author: jkim
 */

#include "PriorBoxLayer.h"
#include "SysLog.h"
#include "MathFunctions.h"

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
	bool adjusted = Layer<Dtype>::_adjustInputShape();

	if (!adjusted)
		return;

	const int layerWidth = this->_inputData[0]->width();
	const int layerHeight = this->_inputData[0]->height();

	/*
	vector<uint32_t> outputShape(4, 1);
	// since all images in a batch has same height and width, we only need to
	// generate one set of priors which can be shared across all images.
	outputShape[0] = 1;
	outputShape[1] = 1;
	// 2 channels. first channel stores the mean of each prior coordinate.
	// second channel stores the variance of each prior coordinate.
	outputShape[2] = 2;
	outputShape[3] = layerWidth * layerHeight * this->numPriors * 4;
	SASSERT0(outputShape[3] > 0);
	*/
	vector<uint32_t> outputShape(4, 1);
	outputShape[0] = 2;
	outputShape[1] = layerWidth * layerHeight * this->numPriors * 4;
	SASSERT0(outputShape[1] > 0);

	this->_outputData[0]->reshape(outputShape);
}

template <typename Dtype>
void PriorBoxLayer<Dtype>::feedforward() {
	reshape();

	const int layerHeight = this->_inputData[0]->height();
	const int layerWidth = this->_inputData[0]->width();
	int imgHeight;
	int imgWidth;
	if (this->imgH == 0 || this->imgW == 0) {
		imgHeight = this->_inputData[1]->height();
		imgWidth = this->_inputData[1]->width();
	} else {
		imgHeight = this->imgH;
		imgWidth = this->imgW;
	}

	Dtype stepH;
	Dtype stepW;
	if (this->stepH == 0 || this->stepW == 0) {
		stepH = Dtype(imgHeight) / layerHeight;
		stepW = Dtype(imgWidth) / layerWidth;
	} else {
		stepH = this->stepH;
		stepW = this->stepW;
	}

	Dtype* outputData = this->_outputData[0]->mutable_host_data();
	int dim = layerHeight * layerWidth * this->numPriors * 4;
	int idx = 0;
	for (int h = 0; h < layerHeight; h++) {
		for (int w = 0; w < layerWidth; w++) {
			Dtype centerX = (w + this->offset) * stepW;
			Dtype centerY = (h + this->offset) * stepH;
			Dtype boxWidth;
			Dtype boxHeight;
			for (int s = 0; s < this->minSizes.size(); s++) {
				int minSize = this->minSizes[s];
				// first prior: aspectRatio = 1, size = minSize
				boxWidth = boxHeight = minSize;
				// xmin
				outputData[idx++] = (centerX - boxWidth / Dtype(2)) / imgWidth;
				// ymin
				outputData[idx++] = (centerY - boxHeight / Dtype(2)) / imgHeight;
				// xmax
				outputData[idx++] = (centerX + boxWidth / Dtype(2)) / imgWidth;
				// ymax
				outputData[idx++] = (centerY + boxHeight / Dtype(2)) / imgHeight;

				if (this->maxSizes.size() > 0) {
					SASSERT0(this->minSizes.size() == this->maxSizes.size());
					int maxSize = this->maxSizes[s];
					// second prior: aspectRatio = 1, size = sqrt(minSize * maxSize)
					boxWidth = boxHeight = sqrt(minSize * maxSize);
					// xmin
					outputData[idx++] = (centerX - boxWidth / Dtype(2)) / imgWidth;
					// ymin
					outputData[idx++] = (centerY - boxHeight / Dtype(2)) / imgHeight;
					// xmax
					outputData[idx++] = (centerX + boxWidth / Dtype(2)) / imgWidth;
					// ymax
					outputData[idx++] = (centerY + boxHeight / Dtype(2)) / imgHeight;
				}

				// rest of priors
				for (int r = 0; r < this->aspectRatios.size(); r++) {
					Dtype ar = this->aspectRatios[r];
					if (fabs(ar - Dtype(1)) < 1e-6) {
						continue;
					}
					boxWidth = minSize * sqrt(ar);
					boxHeight = minSize / sqrt(ar);
					// xmin
					outputData[idx++] = (centerX - boxWidth / Dtype(2)) / imgWidth;
					// ymin
					outputData[idx++] = (centerY - boxHeight / Dtype(2)) / imgHeight;
					// xmax
					outputData[idx++] = (centerX + boxWidth / Dtype(2)) / imgWidth;
					// ymax
					outputData[idx++] = (centerY + boxHeight / Dtype(2)) / imgHeight;
				}
			}
		}
	}
	// clip the prior's coordinate such that it is within [0, 1]
	if (this->clip) {
		for (int d = 0; d < dim; d++) {
			outputData[d] = std::min<Dtype>(std::max<Dtype>(outputData[d], Dtype(0)), Dtype(1));
		}
	}
	// set the variance.
	outputData += this->_outputData[0]->offset(1, 0, 0, 0);
	if (this->variances.size() == 1) {
		soooa_set<Dtype>(dim, Dtype(this->variances[0]), outputData);
	} else {
		int count = 0;
		for (int h = 0; h < layerHeight; h++) {
			for (int w = 0; w < layerWidth; w++) {
				for (int i = 0; i < this->numPriors; i++) {
					for (int j = 0; j < 4; j++) {
						outputData[count] = this->variances[j];
						count++;
					}
				}
			}
		}
	}
}

template <typename Dtype>
void PriorBoxLayer<Dtype>::backpropagation() {
	return;
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





























































