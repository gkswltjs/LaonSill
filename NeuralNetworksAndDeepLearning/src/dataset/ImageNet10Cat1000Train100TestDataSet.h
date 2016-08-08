/**
 * @file	ImageNet10Cat1000Train100TestDataSet.h
 * @date	2016/7/27
 * @author	jhkim
 * @brief
 * @details
 */


#ifndef IMAGENET10CAT1000TRAIN100TESTDATASET_H_
#define IMAGENET10CAT1000TRAIN100TESTDATASET_H_


#include "UbyteDataSet.h"

/**
 * @brief ImageNet 데이터 중 10 카테고리, 1000 학습데이터, 100 테스트데이터 셋을 로드하기 위해
 *        구현된 DataSet 클래스.
 */
class ImageNet10Cat1000Train100TestDataSet : public UbyteDataSet {
public:
	ImageNet10Cat1000Train100TestDataSet()
		: UbyteDataSet(
				"/home/jhkim/image/ILSVRC2012/save/10cat_1000train_100test_100_100/train_data",
				"/home/jhkim/image/ILSVRC2012/save/10cat_1000train_100test_100_100/train_label",
				1,
				"/home/jhkim/image/ILSVRC2012/save/10cat_1000train_100test_100_100/test_data",
				"/home/jhkim/image/ILSVRC2012/save/10cat_1000train_100test_100_100/test_label",
				1,
				3,
				0.8) {

		// ImageNet mean r,g,b 참고 (r,g,b 순서가 뒤집어져있을 수 있음)
		this->mean[0] = 0.407843137;		// 104
		this->mean[1] = 0.454901961;		// 116
		this->mean[2] = 0.478431373;		// 122
	}
	virtual ~ImageNet10Cat1000Train100TestDataSet() {}


	virtual void load() {
		UbyteDataSet::load();
		//numTrainData = 100;
		//numTestData = 100;
	}

};




#endif /* IMAGENET10CAT1000TRAIN100TESTDATASET_H_ */
