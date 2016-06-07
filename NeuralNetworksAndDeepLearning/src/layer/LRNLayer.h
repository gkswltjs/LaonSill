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


class LRNLayer : public HiddenLayer {
public:
	LRNLayer(string name, io_dim in_dim, lrn_dim lrn_d);
	virtual ~LRNLayer();


	rcube &getDeltaInput() { return delta_input; }

	void feedforward(UINT idx, const rcube &input);
	void backpropagation(UINT idx, HiddenLayer *next_layer);

	// update할 weight, bias가 없기 때문에 아래의 method에서는 do nothing
	void reset_nabla(UINT idx) {
		if(!isLastPrevLayerRequest(idx)) throw Exception();
		Layer::reset_nabla(idx);
	}
	void update(UINT idx, double eta, double lambda, int n, int miniBatchSize) {
		if(!isLastPrevLayerRequest(idx)) throw Exception();
		Layer::update(idx, eta, lambda, n, miniBatchSize);
	}

private:
	lrn_dim lrn_d;
	rcube delta_input;
	rcube z;	// beta powered 전의 weighted sum 상태의 norm term

};

#endif /* LAYER_LRNLAYER_H_ */
