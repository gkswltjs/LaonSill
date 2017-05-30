/*
 * LearnableLayer.h
 *
 *  Created on: 2016. 8. 20.
 *      Author: jhkim
 */

#ifndef LEARNABLELAYER_H_
#define LEARNABLELAYER_H_

#include <map>

#include "common.h"
#include "Cuda.h"
#include "Data.h"
#include "BaseLayer.h"

/**
 * @brief 학습하는 레이어에서 구현해야하는 베이스 추상 클래스,
 *        인터페이스 역할을 한다.
 */
template <typename Dtype>
class LearnableLayer : public Layer<Dtype> {
public:
	LearnableLayer();
	virtual ~LearnableLayer() {}


	/**
	 * @details 학습한 파라미터 그레디언트를 파라미터에 업데이트한다.
	 */
	virtual void update() = 0;
	/**
	 * @details 파라미터들의 제곱의 합을 구한다.
	 * @return 파라미터들의 제곱의 합
	 */
	virtual double sumSquareParamsData();

	/**
	 * @details 파라미터 그레디언트들의 제곱의 합을 구한다.
	 * @return 파라미터 그레디언트들의 제곱의 합
	 */
	virtual double sumSquareParamsGrad();

	/**
	 * @details 파라미터 그레디언트를 스케일링한다.
	 * @param 파라미터 그레디언트를 스케일링할 스케일 값
	 */
	virtual void scaleParamsGrad(float scale);
	virtual uint32_t boundParams();
	virtual uint32_t numParams();

	virtual void saveParams(std::ofstream& ofs);
	virtual void loadParams(std::ifstream& ifs);
	virtual void loadParams(std::map<std::string, Data<Dtype>*>& dataMap);

	virtual void applyChanges(LearnableLayer<Dtype> *targetLayer) {}
    virtual void syncParams(LearnableLayer<Dtype> *targetLayer) {}
    virtual void receiveParam(LearnableLayer<Dtype>* donatorLayer) {}

    void fillDonatorInfo(bool isDonator, bool isReceiver, uint32_t donatorID) {
        this->isDonator = isDonator;
        this->isReceiver = isReceiver;
        this->donatorID = donatorID;
    }

protected:
	virtual void _updateParam(const uint32_t paramSize, const Dtype regScale,
                              const Dtype learnScale, Data<Dtype>* dataHistory,
                              Data<Dtype>* data) {};

public:
	std::vector<Data<Dtype>*> _params;
	std::vector<Data<Dtype>*> _paramsHistory;
	std::vector<Data<Dtype>*> _paramsHistory2;
	std::vector<bool> _paramsInitialized;

public:
    bool isDonator;
    bool isReceiver;
    uint32_t donatorID;
};

#endif /* LEARNABLELAYER_H_ */
