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
	DepthConcatLayer() {}
	DepthConcatLayer(const char *name, int n_in);
	DepthConcatLayer(const char *name, io_dim in_dim);
	virtual ~DepthConcatLayer() {}

	rcube &getDeltaInput();



	void feedforward(UINT idx, const rcube &input);

	void backpropagation(UINT idx, HiddenLayer *next_layer);




	void reset_nabla(UINT idx) {
		if(!isLastPrevLayerRequest(idx)) return;
		propResetNParam();
	}
	void update(UINT idx, UINT n, UINT miniBatchSize) {
		if(!isLastPrevLayerRequest(idx)) return;
		propUpdate(n, miniBatchSize);
	}

	void load(ifstream &ifs, map<Layer *, Layer *> &layerMap);


protected:
	void initialize();


	rcube delta_input;
	rcube delta_input_sub;

	vector<int> offsets;
	int offsetIndex;

};

#endif /* LAYER_DEPTHCONCATLAYER_H_ */
