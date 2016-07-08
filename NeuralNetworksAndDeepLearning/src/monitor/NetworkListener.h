/*
 * NetworkListener.h
 *
 *  Created on: 2016. 4. 26.
 *      Author: jhkim
 */

#ifndef MONITOR_NETWORKLISTENER_H_
#define MONITOR_NETWORKLISTENER_H_

class NetworkListener {
public:
	NetworkListener() {}
	virtual ~NetworkListener() {}

	//virtual void epochComplete(double validationCost, double validationAccuracy, double trainCost, double trainAccuracy) = 0;
	virtual void epochComplete(float cost, float accuracy) = 0;
};

#endif /* MONITOR_NETWORKLISTENER_H_ */
