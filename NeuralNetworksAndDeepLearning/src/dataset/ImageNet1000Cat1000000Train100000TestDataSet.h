/*
 * ImageNet1000Cat1000000Train100000TestDataSet.h
 *
 *  Created on: 2016. 7. 27.
 *      Author: jhkim
 */

#ifndef IMAGENET1000CAT1000000TRAIN100000TESTDATASET_H_
#define IMAGENET1000CAT1000000TRAIN100000TESTDATASET_H_



#include "UbyteDataSet.h"

class ImageNet1000Cat1000000Train100000TestDataSet : public UbyteDataSet {
public:
	ImageNet1000Cat1000000Train100000TestDataSet()
		: UbyteDataSet(
				"/home/jhkim/image/ILSVRC2012/save/1000cat_1000000train_100000test_10000_10000/train_data",
				"/home/jhkim/image/ILSVRC2012/save/1000cat_1000000train_100000test_10000_10000/train_label",
				20,
				"/home/jhkim/image/ILSVRC2012/save/1000cat_1000000train_100000test_10000_10000/train_data",
				"/home/jhkim/image/ILSVRC2012/save/1000cat_1000000train_100000test_10000_10000/train_label",
				1,
				3,
				0.8) {

		// ImageNet mean r,g,b 참고 (r,g,b 순서가 뒤집어져있을 수 있음)
		this->mean[0] = 0.407843137;		// 104
		this->mean[1] = 0.454901961;		// 116
		this->mean[2] = 0.478431373;		// 122
	}
	virtual ~ImageNet1000Cat1000000Train100000TestDataSet() {}


	virtual void load() {
		UbyteDataSet::load();
		//numTrainData = 100;
		//numTestData = 100;
	}

};




#endif /* IMAGENET1000CAT1000000TRAIN100000TESTDATASET_H_ */
