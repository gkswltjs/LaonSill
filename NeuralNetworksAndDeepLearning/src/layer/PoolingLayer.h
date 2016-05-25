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


class PoolingLayer : public HiddenLayer {
public:
	PoolingLayer(io_dim in_dim, pool_dim pool_d, Pooling *pooling_fn);
	virtual ~PoolingLayer();

	cube &getDeltaInput() { return this->delta_input; }


	void feedforward(const cube &input);
	void backpropagation(HiddenLayer *next_layer);

	// update할 weight, bias가 없기 때문에 아래의 method에서는 do nothing
	void reset_nabla() {}
	void update(double eta, double lambda, int n, int miniBatchSize) {}

private:
	ucube pool_map;
	cube delta;
	cube delta_input;

	pool_dim pool_d;
	Pooling *pooling_fn;
};

#endif /* LAYER_POOLINGLAYER_H_ */
