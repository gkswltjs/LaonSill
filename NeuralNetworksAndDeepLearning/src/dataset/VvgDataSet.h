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
	}
	virtual ~VvgDataSet() {}

};

#endif /* VVGDATASET_H_ */
