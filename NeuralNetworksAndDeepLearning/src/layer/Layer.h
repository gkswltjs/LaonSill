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

using namespace arma;


class HiddenLayer;


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

	/**
	 * 주어진 입력 input에 대해 출력 activation을 계산
	 * @param input: 레이어 입력 데이터 (이전 레이어의 activation)
	 */
	virtual void feedforward(UINT idx, const rcube &input) {
		for(UINT i = 0; i < nextLayers.size(); i++) {
			nextLayers[i].next_layer->feedforward(nextLayers[i].idx, input);
		}
	}

	virtual void addNextLayer(next_layer_relation nextLayer) { nextLayers.push_back(nextLayer); }


	virtual void reset_nabla(UINT idx) {
		for(unsigned int i = 0; i < nextLayers.size(); i++) {
			nextLayers[i].next_layer->reset_nabla(nextLayers[i].idx);
		}
	}

	virtual void update(UINT idx, int n, int miniBatchSize) {
		for(unsigned int i = 0; i < nextLayers.size(); i++) {
			nextLayers[i].next_layer->update(nextLayers[i].idx, n, miniBatchSize);
		}
	}

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
	vector<bool> forwardFlags;
};

#endif /* LAYER_LAYER_H_ */





























