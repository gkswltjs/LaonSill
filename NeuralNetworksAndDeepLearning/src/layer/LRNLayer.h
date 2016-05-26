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


class LRNLayer : public HiddenLayer {
public:
	LRNLayer(io_dim in_dim, lrn_dim lrn_d);
	virtual ~LRNLayer();


	cube &getDeltaInput() { return delta_input; }

	void feedforward(const cube &input);
	void backpropagation(HiddenLayer *next_layer);

	// update할 weight, bias가 없기 때문에 아래의 method에서는 do nothing
	void reset_nabla() { Layer::reset_nabla(); }
	void update(double eta, double lambda, int n, int miniBatchSize) {
		Layer::update(eta, lambda, n, miniBatchSize);
	}

private:
	lrn_dim lrn_d;
	cube delta_input;
	cube z;	// beta powered 전의 weighted sum 상태의 norm term

};

#endif /* LAYER_LRNLAYER_H_ */
