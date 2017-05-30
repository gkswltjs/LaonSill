/*
 * ConvLayer.cpp
 *
 *  Created on: 2016. 5. 23.
 *      Author: jhkim
 */


#include "ConvLayer.h"
#include "FullyConnectedLayer.h"
#include "Util.h"
#include "Exception.h"
#include "SysLog.h"
#include "PropMgmt.h"

using namespace std;



template<typename Dtype>
void ConvLayer<Dtype>::donateParam(ConvLayer<Dtype>* receiver) {

}

template class ConvLayer<float>;
