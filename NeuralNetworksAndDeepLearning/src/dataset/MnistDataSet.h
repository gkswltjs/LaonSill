/**
 * @file	MnistDataSet.h
 * @date	2016/4/23
 * @author	jhkim
 * @brief
 * @details
 */



#ifndef MNISTDATASET_DATASET_H_
#define MNISTDATASET_DATASET_H_

#include "ImagePackDataSet.h"


/**
 * @brief Mnist 데이터 셋을 로드하기 위해 구현된 DataSet 클래스.
 * @details http://yann.lecun.com/exdb/mnist/의 데이터를 사용.
 * @todo ImagePackDataSet이 channel값을 읽도록 수정되었고, label값을 1000범위까지 읽도록(unsigned int) 수정되어
 *       unsigned char로 label이 저장된 현재의 original 파일을 읽을 경우 error가 발생할 것.
 */
class MnistDataSet : public ImagePackDataSet {
public:
	MnistDataSet(double validationSetRatio)
		: ImagePackDataSet(
				"/home/jhkim/data/learning/mnist/train_data",
				"/home/jhkim/data/learning/mnist/train_label",
				1,
				"/home/jhkim/data/learning/mnist/test_data",
				"/home/jhkim/data/learning/mnist/test_label",
				1) {
		this->mean[0] = 0.13066047740;
	}
	virtual ~MnistDataSet() {}
};

#endif /* MNISTDATASET_DATASET_H_ */
















