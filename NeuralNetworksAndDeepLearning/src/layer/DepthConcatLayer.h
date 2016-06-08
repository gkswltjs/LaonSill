/*
 * DepthConcatLayer.h
 *
 *  Created on: 2016. 5. 25.
 *      Author: jhkim
 */

#ifndef LAYER_DEPTHCONCATLAYER_H_
#define LAYER_DEPTHCONCATLAYER_H_

#include "HiddenLayer.h"
#include "../exception/Exception.h"


class DepthConcatLayer : public HiddenLayer {
public:
	DepthConcatLayer(string name, int n_in);
	DepthConcatLayer(string name, io_dim in_dim);
	virtual ~DepthConcatLayer() {}

	rcube &getDeltaInput();



	void feedforward(UINT idx, const rcube &input);

	void backpropagation(UINT idx, HiddenLayer *next_layer);




	void reset_nabla(UINT idx) {
		if(!isLastPrevLayerRequest(idx)) return;
		Layer::reset_nabla(idx);
	}
	void update(UINT idx, int n, int miniBatchSize) {
		if(!isLastPrevLayerRequest(idx)) return;
		Layer::update(idx, n, miniBatchSize);
	}

protected:
	rcube delta_input;
	rcube delta_input_sub;

	vector<int> offsets;
	int offsetIndex;

};

#endif /* LAYER_DEPTHCONCATLAYER_H_ */
