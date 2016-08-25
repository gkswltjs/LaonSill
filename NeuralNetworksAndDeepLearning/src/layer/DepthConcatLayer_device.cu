/*
 * DepthConcatLayer.cpp
 *
 *  Created on: 2016. 5. 25.
 *      Author: jhkim
 */

#ifdef GPU_MODE

#include "DepthConcatLayer.h"

void DepthConcatLayer::initialize() {
	this->type = Layer::DepthConcat;
	this->offsetIndex = 0;
	this->out_dim.channels = 0;
}


void DepthConcatLayer::_concat(UINT idx, Data* input) {
#ifdef DEPTHCONCAT_LOG
	cout << "depthConcat _concat " << endl;
#endif
	// 이전 레이어들의 전달값을 batch 단위로 합쳐서 다음 레이어로 전달
	// 한 batch내에서의 해당 이전 레이어 전달값의 offset 위치 계산

	int inBatchOffset = 0;
	int i = 0;
	while(i < prevLayers.size() && prevLayers[i]->getId() != idx) {
		inBatchOffset += prevLayers[i]->getOutDimension().unitsize();
		i++;
	}
	io_dim prev_out_dim = prevLayers[i]->getOutDimension();

	//Util::printDeviceData(d_input, in_dim.rows, in_dim.cols, in_dim.channels, in_dim.batches, "d_input:");
	//Util::printDeviceData(input, prev_out_dim.rows, prev_out_dim.cols, prev_out_dim.channels, prev_out_dim.batches, "input:");
	_input->print_data("d_input:");
	input->print_data("input:");

	DATATYPE* d_input = _input->mutable_device_data();
	const DATATYPE* prev_input = input->device_data();
	for(int i = 0; i < prev_out_dim.batches; i++) {
		checkCudaErrors(cudaMemcpyAsync(d_input+in_dim.unitsize()*i+inBatchOffset, prev_input+prev_out_dim.unitsize()*i,
				sizeof(DATATYPE)*prev_out_dim.unitsize(), cudaMemcpyDeviceToDevice));
	}

	//Util::printDeviceData(d_input, in_dim.rows, in_dim.cols, in_dim.channels, in_dim.batches, this->name+string("/d_input:"));
	_input->print_data(this->name+string("/d_input:"));
}


void DepthConcatLayer::_deconcat(UINT idx, Data* next_delta_input, uint32_t offset) {
#ifdef DEPTHCONCAT_LOG
	cout << "depthConcat _deconcat " << endl;
#endif
	if(isFirstNextLayerRequest(idx)) {
		//checkCudaErrors(cudaMemset(d_delta_output, 0, sizeof(DATATYPE)*out_dim.batchsize()));
		_output->reset_device_grad();
		offsetIndex = 0;
	}

	//vector<int> offsets(prevLayers.size());
	//offsets[0] = 0;
	//for(int i = 1; i < offsets.size(); i++) {
	//	offsets[i] = offsets[i-1] + prevLayers[i-1]->getOutDimension().unitsize();
	//}

	//Util::printDeviceData(d_delta_output, out_dim.rows, out_dim.cols, out_dim.channels, out_dim.batches, "d_delta_output:");
	//Util::printDeviceData(next_delta_input, out_dim.rows, out_dim.cols, out_dim.channels, out_dim.batches, "next_delta_input:");
	_output->print_grad("d_delta_output:");
	next_delta_input->print_grad("next_delta_input:");


	const DATATYPE* d_next_delta_input = next_delta_input->device_grad();
	DATATYPE* d_delta_output = _output->mutable_device_grad();
	uint32_t layerOffset = 0;
	for(int j = 0; j < prevLayers.size(); j++) {
		if(j > 0) {
			layerOffset += prevLayers[j-1]->getOutDimension().unitsize();
		}
		for(int i = 0; i < out_dim.batches; i++) {
			//cout << "next_delta_input offset: " << out_dim.unitsize()*i+offsets[j] << ", d_delta_input offset: " << offsets[j]*2+prevLayers[j]->getOutDimension().unitsize()*i << endl;
			checkCudaErrors(cublasSaxpy(
					Cuda::cublasHandle,
					static_cast<int>(prevLayers[j]->getOutDimension().unitsize()),
					&Cuda::alpha,
					d_next_delta_input+out_dim.unitsize()*i+layerOffset,
					1,
					d_delta_output+layerOffset*2+prevLayers[j]->getOutDimension().unitsize()*i,
					1));
		}
	}
	//Util::printDeviceData(d_delta_output, out_dim.rows, out_dim.cols, out_dim.channels, out_dim.batches, "d_delta_output:");
	_output->print_grad("d_delta_output:");
}


#endif



