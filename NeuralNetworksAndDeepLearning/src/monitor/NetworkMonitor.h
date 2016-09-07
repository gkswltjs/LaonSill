/**
 * @file NetworkMonitor.h
 * @date 2016/4/26
 * @author jhkim
 * @brief
 * @details
 */

#ifndef MONITOR_NETWORKMONITOR_H_
#define MONITOR_NETWORKMONITOR_H_

#include <iostream>

#include "NetworkListener.h"
#include <gnuplot-iostream.h>
#include <cmath>
#include <cfloat>
#include <boost/tuple/tuple.hpp>
#include "../debug/GraphPlotter.h"

using namespace std;


/**
 * @brief 네트워크 상태 모니터링 클래스
 * @details 네트워크의 이벤트마다 이벤트 파라미터를 그래프로 ploat하는 역할을 한다.
 */
class NetworkMonitor : public NetworkListener {
public:
	NetworkMonitor()
		: gradSumsqPlotter(500, 10), dataSumsqPlotter(500, 10) {
	}
	virtual ~NetworkMonitor() {}

	void onCostComputed(const uint32_t index, const string name, const double cost) {
		costPlotter.addStat(index, name, cost);
	}
	void onAccuracyComputed(const uint32_t index, const string name, const double accuracy) {
		accuracyPlotter.addStat(index, name, accuracy);
	}
	void onGradSumsqComputed(const uint32_t index, const string name, const double sumsq) {
		gradSumsqPlotter.addStat(index, name, sumsq);
	}
	void onDataSumsqComputed(const uint32_t index, const string name, const double sumsq) {
		dataSumsqPlotter.addStat(index, name, sumsq);
	}


protected:
	GraphPlotter accuracyPlotter;
	GraphPlotter costPlotter;
	GraphPlotter gradSumsqPlotter;
	GraphPlotter dataSumsqPlotter;
};

#endif /* MONITOR_NETWORKMONITOR_H_ */



























