/*
 * LearnableLayer.h
 *
 *  Created on: 2016. 8. 20.
 *      Author: jhkim
 */

#ifndef LEARNABLELAYER_H_
#define LEARNABLELAYER_H_

#include "../Util.h"

class LearnableLayer {
public:
	virtual ~LearnableLayer() {}

	virtual void update() = 0;
	virtual float sumSquareParamsData() = 0;
	virtual float sumSquareParamsGrad() = 0;
	virtual void scaleParamsGrad(DATATYPE scale) = 0;
};



#endif /* LEARNABLELAYER_H_ */
