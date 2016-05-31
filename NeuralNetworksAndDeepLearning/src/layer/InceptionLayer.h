/*
 * InceptionLayer.h
 *
 *  Created on: 2016. 5. 27.
 *      Author: jhkim
 */

#ifndef LAYER_INCEPTIONLAYER_H_
#define LAYER_INCEPTIONLAYER_H_

#include "InputLayer.h"
#include "HiddenLayer.h"


class InceptionLayer : public HiddenLayer {
public:
	InceptionLayer(string name, int n_in, int n_out, int cv1x1, int cv3x3reduce, int cv3x3, int cv5x5reduce, int cv5x5, int p3x3, int cp);
	InceptionLayer(string name, io_dim in_dim, io_dim out_dim, int cv1x1, int cv3x3reduce, int cv3x3, int cv5x5reduce, int cv5x5, int p3x3, int cp);
	virtual ~InceptionLayer();

	cube &getDeltaInput() { return this->delta_input; }

	void feedforward(int idx, const cube &input);
	void backpropagation(int idx, HiddenLayer *next_layer);
	void reset_nabla(int idx);
	void update(int idx, double eta, double lambda, int n, int miniBatchSize);



private:
	void initialize(int cv1x1, int cv3x3reduce, int cv3x3, int cv5x5reduce, int cv5x5, int p3x3, int cp);

	cube delta_input;

	//InputLayer *inputLayer;
	vector<HiddenLayer *> firstLayers;
	HiddenLayer *lastLayer;

};

#endif /* LAYER_INCEPTIONLAYER_H_ */
