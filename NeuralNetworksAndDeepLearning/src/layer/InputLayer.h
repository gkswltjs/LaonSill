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
	InputLayer() {}
	InputLayer(const char *name, io_dim in_dim) : Layer(name, in_dim, in_dim) {
		initialize();
	}
	virtual ~InputLayer() {}

	int getInputDimension() const { return in_dim.rows*in_dim.cols*in_dim.channels; }

	void save(UINT idx, ofstream &ofs) {
		saveHeader(0, ofs);
		// header boundary
		int type = 0;
		Layer *layer = 0;
		ofs.write((char *)&type, sizeof(int));
		ofs.write((char *)&layer, sizeof(Layer *));

		Layer::save(ofs);
		propSave(ofs);
	}

	void load(ifstream &ifs, map<Layer *, Layer *> &layerMap) {
		initialize();

		loadNetwork(ifs, layerMap);
	}

protected:
	void initialize() {
		this->type = LayerType::Input;
		this->id = Layer::generateLayerId();
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
	void feedforward(UINT idx, const rcube &input) {
		//if(!isLastPrevLayerRequest(idx)) throw Exception();

		Util::convertCube(input, this->input);
		Util::convertCube(this->input, this->output);
		Util::printCube(input, "input:");
		Util::printCube(this->output, "output:");

		propFeedforward(this->output);
	}

#else
public:
	void feedforward(UINT idx, const DATATYPE *input) {
		Util::printMessage("InputLayer::feedforward()---"+string(name));
		if(!isLastPrevLayerRequest(idx)) throw Exception();

		Cuda::refresh();

		this->d_input = input;
		//Util::printDeviceData(d_input, in_dim.rows, in_dim.batches, 1, 1, "d_input:");
		checkCudaErrors(cudaMemcpyAsync(this->d_output, this->d_input, sizeof(DATATYPE)*in_dim.batchsize(), cudaMemcpyHostToDevice));

		propFeedforward(this->d_output);
	}
protected:

#endif




};





#endif /* LAYER_INPUTLAYER_H_ */
