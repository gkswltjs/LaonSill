/*
 * PoolingLayer.h
 *
 *  Created on: 2016. 5. 23.
 *      Author: jhkim
 */

#ifndef LAYER_POOLINGLAYER_H_
#define LAYER_POOLINGLAYER_H_

#include "HiddenLayer.h"
#include "../pooling/Pooling.h"
#include "../pooling/PoolingFactory.h"
#include "../exception/Exception.h"



#if CPU_MODE


class PoolingLayer : public HiddenLayer {
public:
	PoolingLayer() {}
	PoolingLayer(const char *name, io_dim in_dim, pool_dim pool_d, PoolingType poolingType);
	virtual ~PoolingLayer();

	rcube &getDeltaInput() { return this->delta_input; }


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
	void initialize(pool_dim pool_d, PoolingType poolingType);
	void save(ofstream &ofs);

	ucube pool_map;
	rcube delta;
	rcube delta_input;

	pool_dim pool_d;
	Pooling *pooling_fn;
};


#else


class PoolingLayer : public HiddenLayer {
public:
	PoolingLayer() {}
	PoolingLayer(const char *name, io_dim in_dim, pool_dim pool_d, PoolingType poolingType);
	virtual ~PoolingLayer();

	rcube &getDeltaInput() { return this->delta_input; }


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
	void initialize(pool_dim pool_d, PoolingType poolingType);
	void save(ofstream &ofs);

	ucube pool_map;
	rcube delta;
	rcube delta_input;

	pool_dim pool_d;
	Pooling *pooling_fn;
};


#endif


#endif /* LAYER_POOLINGLAYER_H_ */
