/*
 * Layer.cu
 *
 *  Created on: 2016. 8. 25.
 *      Author: jhkim
 */

#ifdef GPU_MODE

#include "Layer.h"
#include "Cuda.h"


using namespace std;

template <typename Dtype>
void Layer<Dtype>::_clearShape() {
	//checkCUDNN(cudnnDestroyTensorDescriptor(inputTensorDesc));
	//checkCUDNN(cudnnDestroyTensorDescriptor(outputTensorDesc));

	//delete _input;
	//delete _output;
	//_input = NULL;
	//_output = NULL;
	//inputTensorDesc = NULL;
	//outputTensorDesc = NULL;
}

//template void Layer<float>::_shape(bool recursive);
template void Layer<float>::_clearShape();

#endif



































