/*
 * Layer.h
 *
 *  Created on: 2016. 5. 10.
 *      Author: jhkim
 */

#ifndef LAYER_LAYER_H_
#define LAYER_LAYER_H_

#include "LayerConfig.h"
#include "../Util.h"
#include <armadillo>
#include <iostream>

using namespace arma;


enum class LayerType {
	Input, FullyConnected, Conv, Pooling, DepthConcat, Inception, LRN, Sigmoid, Softmax
};



class Layer {
public:
	Layer(string name, int n_in, int n_out) {
		this->name = name;
		this->in_dim.rows = n_in;
		this->out_dim.rows = n_out;
		this->input.set_size(n_in, 1, 1);
		this->output.set_size(n_out, 1, 1);
	}
	Layer(string name, io_dim in_dim, io_dim out_dim) {
		this->name = name;
		this->in_dim = in_dim;
		this->out_dim = out_dim;
		this->input.set_size(in_dim.rows, in_dim.cols, in_dim.channels);
		this->output.set_size(out_dim.rows, out_dim.cols, out_dim.channels);
	}
	virtual ~Layer() {}

	rcube &getInput() { return this->input; }
	rcube &getOutput() { return this->output; }
	vector<next_layer_relation> &getNextLayers() { return this->nextLayers; }
	int getNextLayerSize() { return this->nextLayers.size(); }
	void addNextLayer(next_layer_relation nextLayer) { nextLayers.push_back(nextLayer); }


	/**
	 * 주어진 입력 input에 대해 출력 activation을 계산
	 * @param input: 레이어 입력 데이터 (이전 레이어의 activation)
	 */
	virtual void feedforward(UINT idx, const rcube &input) { propFeedforward(input); }
	virtual void reset_nabla(UINT idx) { propResetNParam(); }
	virtual void update(UINT idx, UINT n, UINT miniBatchSize) { propUpdate(n, miniBatchSize); }
	virtual void save(UINT idx, ofstream &ofs) { propSave(ofs); }


protected:

	bool isLastPrevLayerRequest(UINT idx) {
		//cout << name << " received request from " << idx << "th prev layer ... " << endl;
		if(prevLayers.size() > idx+1) {
			//cout << name << " is not from last prev layer... " << endl;
			return false;
		} else {
			return true;
		}
	}

	void propFeedforward(const rcube output) {
		for(UINT i = 0; i < nextLayers.size(); i++) {
			nextLayers[i].next_layer->feedforward(nextLayers[i].idx, output);
		}
	}

	void propResetNParam() {
		for(UINT i = 0; i < nextLayers.size(); i++) {
			nextLayers[i].next_layer->reset_nabla(nextLayers[i].idx);
		}
	}

	void propUpdate(UINT n, UINT miniBatchSize) {
		for(UINT i = 0; i < nextLayers.size(); i++) {
			nextLayers[i].next_layer->update(nextLayers[i].idx, n, miniBatchSize);
		}
	}

	void propSave(ofstream &ofs) {
		for(UINT i = 0; i < nextLayers.size(); i++) {
			nextLayers[i].next_layer->save(nextLayers[i].idx, ofs);
		}
	}


	string name;


	io_dim in_dim;
	io_dim out_dim;

	/**
	 * activation이자 레이어의 output
	 */
	rcube input;
	rcube output;

	vector<prev_layer_relation> prevLayers;
	vector<next_layer_relation> nextLayers;


	LayerType type;

};

#endif /* LAYER_LAYER_H_ */





























