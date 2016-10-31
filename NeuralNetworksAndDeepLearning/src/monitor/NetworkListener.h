/**
 * @file NetworkListener.h
 * @date 2016/4/26
 * @author jhkim
 * @brief
 * @details
 */




#ifndef MONITOR_NETWORKLISTENER_H_
#define MONITOR_NETWORKLISTENER_H_

#include "../common.h"

/**
 * @brief 네트워크 내부 이벤트 리스너 기본 추상 클래스
 * @details 네트워크 내부에서 발생한 이벤트와 해당 이벤트의 파라미터값을 갖고 필요에 따라 적절한 수행을 한다.
 */
class NetworkListener {
public:
	NetworkListener() {}
	virtual ~NetworkListener() {}

	//virtual void epochComplete(double validationCost, double validationAccuracy, double trainCost, double trainAccuracy) = 0;
	/**
	 * @details epoch 종료 이벤트 리스너
	 * @param cost epoch에서 계산된 cost값
	 * @param accuracy epoch에서 계산된 정확도값
	 */
	//virtual void epochComplete(float cost, float accuracy) = 0;

	virtual void onCostComputed(const uint32_t index, const std::string name, const double cost) = 0;
	virtual void onAccuracyComputed(const uint32_t index, const std::string name, const double accuracy) = 0;
	virtual void onGradSumsqComputed(const uint32_t index, const std::string name, const double sumq) = 0;
	virtual void onDataSumsqComputed(const uint32_t index, const std::string name, const double sumq) = 0;


};

#endif /* MONITOR_NETWORKLISTENER_H_ */
