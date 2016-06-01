/*
 * GoogLeNetMnist.h
 *
 *  Created on: 2016. 6. 1.
 *      Author: jhkim
 */

#ifndef NETWORK_GOOGLENETMNIST_H_
#define NETWORK_GOOGLENETMNIST_H_

#include "Network.h"


class GoogLeNetMnist : public Network {
public:
	GoogLeNetMnist(NetworkListener *networkListener);
	virtual ~GoogLeNetMnist();
};

#endif /* NETWORK_GOOGLENETMNIST_H_ */
