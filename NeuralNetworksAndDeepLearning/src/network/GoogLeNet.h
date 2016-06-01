/*
 * GoogLeNet.h
 *
 *  Created on: 2016. 5. 31.
 *      Author: jhkim
 */

#ifndef NETWORK_GOOGLENET_H_
#define NETWORK_GOOGLENET_H_

#include "Network.h"


class GoogLeNet : public Network {
public:
	GoogLeNet(NetworkListener *networkListener);
	virtual ~GoogLeNet();
};

#endif /* NETWORK_GOOGLENET_H_ */
