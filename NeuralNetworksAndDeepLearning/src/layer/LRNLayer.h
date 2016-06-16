/*
 * LRNLayer.h
 *
 *  Created on: 2016. 5. 25.
 *      Author: jhkim
 */

#ifndef LAYER_LRNLAYER_H_
#define LAYER_LRNLAYER_H_

#include "HiddenLayer.h"
#include "LayerConfig.h"
#include "../exception/Exception.h"



#if CPU_MODE


class LRNLayer : public HiddenLayer {
public:
	LRNLayer() {}
	LRNLayer(const char *name, io_dim in_dim, lrn_dim lrn_d);
	virtual ~LRNLayer();


	rcube &getDeltaInput() { return delta_input; }

	void feedforward(UINT idx, const rcube &input);
	void backpropagation(UINT idx, HiddenLayer *next_layer);

	// update할 weight, bias가 없기 때문에 아래의 method에서는 do nothing
	void reset_nabla(UINT idx) {
		if(!isLastPrevLayerRequest(idx)) throw Exception();
		propResetNParam();
	}
	void update(UINT idx, UINT n, UINT miniBatchSize) {
		if(!isLastPrevLayerRequest(idx)) throw Exception();
		propUpdate(n, miniBatchSize);
	}

	void save(UINT idx, ofstream &ofs);
	void load(ifstream &ifs, map<Layer *, Layer *> &layerMap);

private:
	void initialize(lrn_dim lrn_d);
	void save(ofstream &ofs);

	lrn_dim lrn_d;
	rcube delta_input;
	rcube z;	// beta powered 전의 weighted sum 상태의 norm term

};


#else


class LRNLayer : public HiddenLayer {
public:
	LRNLayer() {}
	LRNLayer(const char *name, io_dim in_dim, lrn_dim lrn_d);
	virtual ~LRNLayer();


	rcube &getDeltaInput() { return delta_input; }

	void feedforward(UINT idx, const rcube &input);
	void backpropagation(UINT idx, HiddenLayer *next_layer);

	// update할 weight, bias가 없기 때문에 아래의 method에서는 do nothing
	void reset_nabla(UINT idx) {
		if(!isLastPrevLayerRequest(idx)) throw Exception();
		propResetNParam();
	}
	void update(UINT idx, UINT n, UINT miniBatchSize) {
		if(!isLastPrevLayerRequest(idx)) throw Exception();
		propUpdate(n, miniBatchSize);
	}

	void save(UINT idx, ofstream &ofs);
	void load(ifstream &ifs, map<Layer *, Layer *> &layerMap);

private:
	void initialize(lrn_dim lrn_d);
	void save(ofstream &ofs);

	lrn_dim lrn_d;
	rcube delta_input;
	rcube z;	// beta powered 전의 weighted sum 상태의 norm term

};


#endif



#endif /* LAYER_LRNLAYER_H_ */
