/**
 * @file	VvgDataSet.h
 * @date	2016/7/14
 * @author	jhkim
 * @brief
 * @details
 */






#ifndef VVGDATASET_H_
#define VVGDATASET_H_

#include "UbyteDataSet.h"


/**
 * @brief 반고흐 이미지 데이터셋을 로드하기 위해 DataSet을 구현한 클래스.
 * @details
 * @todo 데이터셋의 포맷이 구버전이기 때문에 사용하기 위해서는 데이터셋의 포맷을 갱신해야 한다.
 */
class VvgDataSet : public UbyteDataSet {
public:
	VvgDataSet(double validationSetRatio)
		: UbyteDataSet(
				"/home/jhkim/data/learning/vvg/vvg_image.ubyte",
				"/home/jhkim/data/learning/vvg/vvg_label.ubyte",
				1,
				"/home/jhkim/data/learning/vvg/vvg_image.ubyte",
				"/home/jhkim/data/learning/vvg/vvg_label.ubyte",
				1,
				3,
				validationSetRatio) {
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
