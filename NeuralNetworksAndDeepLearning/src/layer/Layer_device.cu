/*
 * Layer.cu
 *
 *  Created on: 2016. 8. 25.
 *      Author: jhkim
 */


#ifdef GPU_MODE

#include "Layer.h"
#include "../cuda/Cuda.h"
#include <iostream>


void Layer::_shape(bool recursive) {
	char message[256];
	sprintf(message, "%s---_shape():in-%dx%dx%dx%d, out-%dx%dx%dx%d", name.c_str(), in_dim.rows, in_dim.cols, in_dim.channels, in_dim.batches,
			out_dim.rows, out_dim.cols, out_dim.channels, out_dim.batches);
	Util::setPrint(true);
	Util::printMessage(string(message));
	Util::setPrint(false);

	//checkCudaErrors(Util::ucudaMalloc(&this->d_input, sizeof(DATATYPE)*in_dim.batchsize()));		//batch size 고려
	//checkCudaErrors(Util::ucudaMalloc(&this->d_output, sizeof(DATATYPE)*out_dim.batchsize()));		//batch size 고려
	_input->reshape({in_dim.batches, in_dim.channels, in_dim.rows, in_dim.cols});
	_output->reshape({out_dim.batches, out_dim.channels, out_dim.rows, out_dim.cols});

	checkCUDNN(cudnnCreateTensorDescriptor(&inputTensorDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&outputTensorDesc));

	checkCUDNN(cudnnSetTensor4dDescriptor(inputTensorDesc,
			CUDNN_TENSOR_NCHW,
			CUDNN_DATA_FLOAT,
			in_dim.batches, in_dim.channels, in_dim.rows, in_dim.cols));
	checkCUDNN(cudnnSetTensor4dDescriptor(outputTensorDesc,
			CUDNN_TENSOR_NCHW,
			CUDNN_DATA_FLOAT,
			out_dim.batches, out_dim.channels, out_dim.rows, out_dim.cols));
}

void Layer::_clearShape() {
	//checkCudaErrors(cudaFree(d_input));
	//checkCudaErrors(cudaFree(d_output));
	checkCUDNN(cudnnDestroyTensorDescriptor(inputTensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(outputTensorDesc));

	delete _input;
	delete _output;
	_input = NULL;
	_output = NULL;
	inputTensorDesc = NULL;
	outputTensorDesc = NULL;
}


void Layer::_save(ofstream &ofs) {
	Layer *address = this;
	UINT nextLayerSize = nextLayers.size();
	UINT prevLayerSize = prevLayers.size();

	ofs.write((char *)&address, sizeof(Layer *));							// layer address
	ofs.write((char *)&id, sizeof(int));									// layer id
	ofs.write(name.c_str(), LAYER_NAME_LENGTH);								// layer name
	ofs.write((char *)&in_dim, sizeof(io_dim));								// layer in_dim
	ofs.write((char *)&out_dim, sizeof(io_dim));							// layer out_dim
	ofs.write((char *)&nextLayerSize, sizeof(UINT));						// layer next layer size
	for(UINT i = 0; i < nextLayerSize; i++) {								// layer next layers
		ofs.write((char *)&nextLayers[i], sizeof(Layer*));
	}
	ofs.write((char *)&prevLayerSize, sizeof(UINT));						// layer prev layer size
	for(UINT i = 0; i < prevLayers.size(); i++) {							// layer prev layers
		ofs.write((char *)&prevLayers[i], sizeof(Layer*));
	}
}

void Layer::_load(ifstream &ifs, map<Layer *, Layer *> &layerMap) {
	int layerId;
	char name[LAYER_NAME_LENGTH];
	UINT nextLayerSize, prevLayerSize;

	ifs.read((char *)&layerId, sizeof(int));
	ifs.read(name, LAYER_NAME_LENGTH);
	ifs.read((char *)&in_dim, sizeof(io_dim));
	ifs.read((char *)&out_dim, sizeof(io_dim));
	ifs.read((char *)&nextLayerSize, sizeof(UINT));
	for(UINT i = 0; i < nextLayerSize; i++) {
		Layer* nextLayer;
		ifs.read((char *)&nextLayer, sizeof(Layer*));
		nextLayers.push_back(nextLayer);
	}
	ifs.read((char *)&prevLayerSize, sizeof(UINT));
	for(UINT i = 0; i < prevLayerSize; i++) {
		Layer* prevLayer;
		ifs.read((char *)&prevLayer, sizeof(Layer*));
		prevLayers.push_back(prevLayer);
	}
	initialize(layerId, name);

	Layer::_shape(false);
	updateLayerRelation(layerMap);
}


//template void Layer::_shape(bool recursive);
//template void Layer::_clearShape();
//template void Layer::_save(ofstream &ofs);
//template void Layer::_load(ifstream &ifs, map<Layer *, Layer *> &layerMap);


#endif



































