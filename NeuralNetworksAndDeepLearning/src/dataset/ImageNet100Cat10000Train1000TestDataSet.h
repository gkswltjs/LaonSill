/*
 * ImageNet100Cat10000Train1000TestDataSet.h
 *
 *  Created on: 2016. 7. 27.
 *      Author: jhkim
 */

#ifndef IMAGENET100CAT10000TRAIN1000TESTDATASET_H_
#define IMAGENET100CAT10000TRAIN1000TESTDATASET_H_

#include "UbyteDataSet.h"

class ImageNet100Cat10000Train1000TestDataSet : public UbyteDataSet {
public:
	ImageNet100Cat10000Train1000TestDataSet()
		: UbyteDataSet(
				//"/home/jhkim/image/ILSVRC2012/save/train_data",
				//"/home/jhkim/image/ILSVRC2012/save/train_label",
				"/home/jhkim/image/ILSVRC2012/save/100cat_10000train_1000test_10000_1000/train_data",
				"/home/jhkim/image/ILSVRC2012/save/100cat_10000train_1000test_10000_1000/train_label",
				1,
				"/home/jhkim/image/ILSVRC2012/save/100cat_10000train_1000test_10000_1000/train_data",
				"/home/jhkim/image/ILSVRC2012/save/100cat_10000train_1000test_10000_1000/train_label",
				//"/home/jhkim/image/ILSVRC2012/save/100cat_10000train_1000test_10000_1000/test_data",
				//"/home/jhkim/image/ILSVRC2012/save/100cat_10000train_1000test_10000_1000/test_label",
				1,
				3,
				0.8) {

		// ImageNet mean r,g,b 참고 (r,g,b 순서가 뒤집어져있을 수 있음)
		this->mean[0] = 0.47684615850;		// R: 122
		this->mean[1] = 0.45469805598;		// G: 116
		this->mean[2] = 0.41394191980;		// B: 104
	}
	virtual ~ImageNet100Cat10000Train1000TestDataSet() {}


	virtual void load() {
		UbyteDataSet::load();
		//numTrainData = 100;
		//numTestData = 100;
	}

};


#endif /* IMAGENET100CAT10000TRAIN1000TESTDATASET_H_ */
