/*
 * NetworkMonitor.h
 *
 *  Created on: 2016. 4. 26.
 *      Author: jhkim
 */

#ifndef MONITOR_NETWORKMONITOR_H_
#define MONITOR_NETWORKMONITOR_H_

#include <iostream>

#include "NetworkListener.h"

using namespace std;


class NetworkMonitor : public NetworkListener {
public:
	NetworkMonitor() {}
	virtual ~NetworkMonitor() {}

	void epochComplete(double validationCost, double validationAccuracy, double trainCost, double trainAccuracy) {
		cout << "vc: " << validationCost << ", va: " << validationAccuracy << ", tc: " << trainCost << ", ta: " << trainAccuracy << endl;
	}
};

#endif /* MONITOR_NETWORKMONITOR_H_ */
