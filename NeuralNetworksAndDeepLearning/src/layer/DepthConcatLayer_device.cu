/*
 * DepthConcatLayer.cpp
 *
 *  Created on: 2016. 5. 25.
 *      Author: jhkim
 */


#ifdef GPU_MODE

#include "DepthConcatLayer.h"

template <typename Dtype>
void DepthConcatLayer<Dtype>::initialize() {
	this->type = Layer<Dtype>::DepthConcat;
	this->offsetIndex = 0;
	this->out_dim.channels = 0;
}


template <typename Dtype>
void DepthConcatLayer<Dtype>::_concat(uint32_t idx, Data<Dtype>* input) {
#ifdef DEPTHCONCAT_LOG
	cout << "depthConcat _concat " << endl;
#endif
	// 이전 레이어들의 전달값을 batch 단위로 합쳐서 다음 레이어로 전달
	// 한 batch내에서의 해당 이전 레이어 전달값의 offset 위치 계산

	int inBatchOffset = 0;
	int i = 0;
	while(i < this->prevLayers.size() && this->prevLayers[i]->getId() != idx) {
		inBatchOffset += this->prevLayers[i]->getOutDimension().unitsize();
		i++;
	}
	io_dim prev_out_dim = this->prevLayers[i]->getOutDimension();

	//Util::printDeviceData(d_input, in_dim.rows, in_dim.cols, in_dim.channels, in_dim.batches, "d_input:");
	//Util::printDeviceData(input, prev_out_dim.rows, prev_out_dim.cols, prev_out_dim.channels, prev_out_dim.batches, "input:");
	this->_input->print_data("d_input:");
	input->print_data("input:");

	Dtype* d_input = this->_input->mutable_device_data();
	const Dtype* prev_input = input->device_data();
	for(int i = 0; i < prev_out_dim.batches; i++) {
		checkCudaErrors(cudaMemcpyAsync(d_input+this->in_dim.unitsize()*i+inBatchOffset, prev_input+prev_out_dim.unitsize()*i,
				sizeof(Dtype)*prev_out_dim.unitsize(), cudaMemcpyDeviceToDevice));
	}

	//Util::printDeviceData(d_input, in_dim.rows, in_dim.cols, in_dim.channels, in_dim.batches, this->name+string("/d_input:"));
	this->_input->print_data(this->name+string("/d_input:"));
}


template <typename Dtype>
void DepthConcatLayer<Dtype>::_deconcat(uint32_t idx, Data<Dtype>* next_delta_input, uint32_t offset) {
#ifdef DEPTHCONCAT_LOG
	cout << "depthConcat _deconcat " << endl;
#endif
	if(this->isFirstNextLayerRequest(idx)) {
		//checkCudaErrors(cudaMemset(d_delta_output, 0, sizeof(Dtype)*out_dim.batchsize()));
		this->_output->reset_device_grad();
		offsetIndex = 0;
	}

	//vector<int> offsets(prevLayers.size());
	//offsets[0] = 0;
	//for(int i = 1; i < offsets.size(); i++) {
	//	offsets[i] = offsets[i-1] + prevLayers[i-1]->getOutDimension().unitsize();
	//}

	//Util::printDeviceData(d_delta_output, out_dim.rows, out_dim.cols, out_dim.channels, out_dim.batches, "d_delta_output:");
	//Util::printDeviceData(next_delta_input, out_dim.rows, out_dim.cols, out_dim.channels, out_dim.batches, "next_delta_input:");
	this->_output->print_grad("d_delta_output:");
	next_delta_input->print_grad("next_delta_input:");


	const Dtype* d_next_delta_input = next_delta_input->device_grad();
	Dtype* d_delta_output = this->_output->mutable_device_grad();
	uint32_t layerOffset = 0;
	for(int j = 0; j < this->prevLayers.size(); j++) {
		if(j > 0) {
			layerOffset += this->prevLayers[j-1]->getOutDimension().unitsize();
		}
		for(int i = 0; i < this->out_dim.batches; i++) {
			//cout << "next_delta_input offset: " << out_dim.unitsize()*i+offsets[j] << ", d_delta_input offset: " << offsets[j]*2+prevLayers[j]->getOutDimension().unitsize()*i << endl;
			checkCudaErrors(cublasSaxpy(
					Cuda::cublasHandle,
					static_cast<int>(this->prevLayers[j]->getOutDimension().unitsize()),
					&Cuda::alpha,
					d_next_delta_input+this->out_dim.unitsize()*i+layerOffset,
					1,
					d_delta_output+layerOffset*2+this->prevLayers[j]->getOutDimension().unitsize()*i,
					1));
		}
	}
	//Util::printDeviceData(d_delta_output, out_dim.rows, out_dim.cols, out_dim.channels, out_dim.batches, "d_delta_output:");
	this->_output->print_grad("d_delta_output:");
}


template void DepthConcatLayer<float>::initialize();
template void DepthConcatLayer<float>::_concat(uint32_t idx, Data<float>* input);
template void DepthConcatLayer<float>::_deconcat(uint32_t idx, Data<float>* next_delta_input, uint32_t offset);


#endif


