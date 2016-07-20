/*
 * VvgDataSet.h
 *
 *  Created on: 2016. 7. 14.
 *      Author: jhkim
 */

#ifndef VVGDATASET_H_
#define VVGDATASET_H_

#include "UbyteDataSet.h"

class VvgDataSet : public UbyteDataSet {
public:
	VvgDataSet(double validationSetRatio)
		:	UbyteDataSet(
				"/home/jhkim/data/learning/vvg/vvg_image.ubyte",
				"/home/jhkim/data/learning/vvg/vvg_label.ubyte",
				"/home/jhkim/data/learning/vvg/vvg_image.ubyte",
				"/home/jhkim/data/learning/vvg/vvg_label.ubyte",
				validationSetRatio) {
		this->channels = 3;

		this->mean[0] = 0.55806416406;
		this->mean[1] = 0.51419142615;
		this->mean[2] = 0.40562924818;
	}
	virtual ~VvgDataSet() {}


	virtual void load() {
		UbyteDataSet::load();
		//numTrainData = 100;
		//numTestData = 100;
	}

};

#endif /* VVGDATASET_H_ */
