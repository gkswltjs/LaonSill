/*
 * ImageNet10Cat100Train100TestDataSet.h
 *
 *  Created on: 2016. 7. 27.
 *      Author: jhkim
 */

#ifndef IMAGENET10CAT100TRAIN100TESTDATASET_H_
#define IMAGENET10CAT100TRAIN100TESTDATASET_H_


#include "UbyteDataSet.h"

class ImageNet10Cat100Train100TestDataSet : public UbyteDataSet {
public:
	ImageNet10Cat100Train100TestDataSet()
		: UbyteDataSet(
				"/home/jhkim/image/ILSVRC2012/save/10cat_100train_100test_100_100/train_data",
				"/home/jhkim/image/ILSVRC2012/save/10cat_100train_100test_100_100/train_label",
				1,
				"/home/jhkim/image/ILSVRC2012/save/10cat_100train_100test_100_100/test_data",
				"/home/jhkim/image/ILSVRC2012/save/10cat_100train_100test_100_100/test_label",
				1,
				3,
				0.8) {

		// ImageNet mean r,g,b 참고 (r,g,b 순서가 뒤집어져있을 수 있음)
		this->mean[0] = 0.407843137;		// 104
		this->mean[1] = 0.454901961;		// 116
		this->mean[2] = 0.478431373;		// 122
	}
	virtual ~ImageNet10Cat100Train100TestDataSet() {}


	virtual void load() {
		UbyteDataSet::load();
		//numTrainData = 100;
		//numTestData = 100;

		//rows = 7;
		//cols = 7;
		//channels = 1;
	}

};




#endif /* IMAGENET10CAT100TRAIN100TESTDATASET_H_ */
