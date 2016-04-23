/*
 * activation.cpp
 *
 *  Created on: 2016. 4. 21.
 *      Author: jhkim
 */

#include "activation.h"






vec sigmoid(vec &activation) {
	return arma::exp(activation);
}
