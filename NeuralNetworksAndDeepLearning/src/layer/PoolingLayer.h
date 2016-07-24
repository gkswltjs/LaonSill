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






class PoolingLayer : public HiddenLayer {
public:
	PoolingLayer() { this->type = LayerType::Pooling; }
	PoolingLayer(const char *name, pool_dim pool_d, PoolingType poolingType);
	virtual ~PoolingLayer();

	void backpropagation(UINT idx, DATATYPE *next_delta_input);

	// update할 weight, bias가 없기 때문에 아래의 method에서는 do nothing
	void reset_nabla(UINT idx) {
		if(!isLastPrevLayerRequest(idx)) throw Exception();
		propResetNParam();
	}
	void update(UINT idx, UINT n, UINT miniBatchSize) {
		if(!isLastPrevLayerRequest(idx)) throw Exception();
		propUpdate(n, miniBatchSize);
	}

	void load(ifstream &ifs, map<Layer *, Layer *> &layerMap);

#if CPU_MODE
public:
	rcube &getDeltaInput() { return this->delta_input; }
	void feedforward(UINT idx, const rcube &input, const char *end=0);

#else
public:
	DATATYPE *getDeltaInput() { return this->d_delta_input; }

	void feedforward(UINT idx, const DATATYPE *input, const char *end=0);
#endif

protected:
	void initialize(pool_dim pool_d, PoolingType poolingType);
	virtual void _save(ofstream &ofs);
	virtual void _shape(bool recursive=true);
	virtual void _clearShape();

	pool_dim pool_d;
	Pooling *pooling_fn;

#if CPU_MODE
protected:
	ucube pool_map;
	rcube delta;
	rcube delta_input;
#else
protected:
	DATATYPE *d_delta;
#endif

};





#endif /* LAYER_POOLINGLAYER_H_ */
