/*
 * InputLayer.h
 *
 *  Created on: 2016. 5. 11.
 *      Author: jhkim
 */

#ifndef LAYER_INPUTLAYER_H_
#define LAYER_INPUTLAYER_H_

#include "Layer.h"
#include "LayerConfig.h"
#include "../Util.h"
#include "../exception/Exception.h"
#include <armadillo>

using namespace arma;


class InputLayer : public Layer {
public:
	InputLayer() { this->type = LayerType::Input; }
	InputLayer(const char *name) : Layer(name) {
		initialize();
	}
	virtual ~InputLayer() {}


	int getInputDimension() const { return in_dim.rows*in_dim.cols*in_dim.channels; }

	virtual void save(UINT idx, ofstream &ofs) {
		saveHeader(0, ofs);

		// header boundary (dummy layer)
		int type = 0;
		Layer *layer = 0;
		ofs.write((char *)&type, sizeof(int));
		ofs.write((char *)&layer, sizeof(Layer *));

		Layer::_save(ofs);
		propSave(ofs);
	}

	virtual void load(ifstream &ifs, map<Layer *, Layer *> &layerMap) {
		initialize();
		InputLayer::_shape(false);
		loadNetwork(ifs, layerMap);
	}

#if CPU_MODE
public:
	InputLayer(const char *name, int n_in) : Layer(name, n_in, n_in) {
		initialize();
	}
	/**
	 * Input 무조건 첫번째 layer,
	 * feedforward로 들어오는 input외의 input에 대해서는 고려하지 않음
	 */
	void feedforward(UINT idx, const rcube &input, const char *end=0) {
		//if(!isLastPrevLayerRequest(idx)) throw Exception();

		Util::convertCube(input, this->input);
		Util::convertCube(this->input, this->output);
		Util::printCube(input, "input:");
		Util::printCube(this->output, "output:");

		propFeedforward(this->output, end);
	}

#else
public:
	void feedforward(UINT idx, const DATATYPE *input, const char *end=0) {
		Util::printMessage("InputLayer::feedforward()---"+string(name));
		if(!isLastPrevLayerRequest(idx)) throw Exception();

		Cuda::refresh();

		this->d_input = input;
		//Util::printDeviceData(d_input, in_dim.rows, in_dim.batches, 1, 1, "d_input:");
		checkCudaErrors(cudaMemcpyAsync(this->d_output, this->d_input, sizeof(DATATYPE)*in_dim.batchsize(), cudaMemcpyDeviceToDevice));

		propFeedforward(this->d_output, end);
	}
#endif




protected:
	void initialize() {
		this->type = LayerType::Input;
	}

#if CPU_MODE
protected:
#else
protected:
	virtual void _shape(bool recursive=true) {
		this->out_dim = in_dim;
		if(recursive) {
			Layer::_shape();
		}
	}

	virtual void _clearShape() {
		Layer::_clearShape();
	}

#endif




};





#endif /* LAYER_INPUTLAYER_H_ */
