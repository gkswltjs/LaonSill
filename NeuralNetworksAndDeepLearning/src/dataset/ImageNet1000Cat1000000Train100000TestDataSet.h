/**
 * @file	ImageNet1000Cat1000000Train100000TestDataSet.h
 * @date	2016/7/27
 * @author	jhkim
 * @brief
 * @details
 */

#ifndef IMAGENET1000CAT1000000TRAIN100000TESTDATASET_H_
#define IMAGENET1000CAT1000000TRAIN100000TESTDATASET_H_



#include "UbyteDataSet.h"


/**
 * @brief ImageNet 데이터 중 1,000 카테고리, 1,000,000 학습데이터, 100,000 테스트데이터 셋을 로드하기 위해
 *        구현된 DataSet 클래스.
 * @details 1,000,000 학습 데이터는 50,000장의 이미지로 구성된 데이터 파일 20개로 구성, 각 데이터 파일은 약 7GB
 *          메모리로 로드할 경우 RealData로 변환되어 float인 경우 약 28GB, Double의 경우 56GB의 메모리 사용.
 */
class ImageNet1000Cat1000000Train100000TestDataSet : public UbyteDataSet {
public:
	ImageNet1000Cat1000000Train100000TestDataSet()
		: UbyteDataSet(
				"/home/jhkim/image/ILSVRC2012/save/1000cat_1000000train_10000test_50000_10000/train_data",
				"/home/jhkim/image/ILSVRC2012/save/1000cat_1000000train_10000test_50000_10000/train_label",
				20,
				"/home/jhkim/image/ILSVRC2012/save/1000cat_1000000train_10000test_50000_10000/test_data",
				"/home/jhkim/image/ILSVRC2012/save/1000cat_1000000train_10000test_50000_10000/test_label",
				1,
				3,
				0.8) {

		// ImageNet mean r,g,b 참
		this->mean[0] = 122.0 / 255.0;		// R: 122
		this->mean[1] = 116.0 / 255.0;		// G: 116
		this->mean[2] = 104.0 / 255.0;		// B: 104
	}
	virtual ~ImageNet1000Cat1000000Train100000TestDataSet() {}

	virtual void load() {
		UbyteDataSet::load();
		//numTrainData = 100;
		//numTestData = 100;
	}

};




#endif /* IMAGENET1000CAT1000000TRAIN100000TESTDATASET_H_ */
