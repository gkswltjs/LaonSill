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
	InceptionLayer() { this->type = LayerType::Inception; }
	InceptionLayer(const char *name, int ic, int oc_cv1x1, int oc_cv3x3reduce, int oc_cv3x3, int oc_cv5x5reduce, int oc_cv5x5, int oc_cp);
	virtual ~InceptionLayer();

	void backpropagation(UINT idx, HiddenLayer *next_layer);
	void reset_nabla(UINT idx);
	void update(UINT idx, UINT n, UINT miniBatchSize);

	void load(ifstream &ifs, map<Layer *, Layer *> &layerMap);
	void saveNinHeader(UINT idx, ofstream &ofs);

#if CPU_MODE
public:
	InceptionLayer(const char *name, int n_in, int n_out, int cv1x1, int cv3x3reduce, int cv3x3, int cv5x5reduce, int cv5x5, int cp);
	rcube &getDeltaInput() { return this->delta_input; }
	void feedforward(UINT idx, const rcube &input);
#else
public:
	DATATYPE *getDeltaInput() { return this->d_delta_input; }
	void feedforward(UINT idx, const DATATYPE *input);
#endif

protected:
	void initialize();
	void initialize(int ic, int cv1x1, int cv3x3reduce, int cv3x3, int cv5x5reduce, int cv5x5, int cp);

	virtual void _save(ofstream &ofs);
	virtual void _shape();
	virtual void _reshape();
	virtual void _clearShape();

	//InputLayer *inputLayer;
	vector<HiddenLayer *> firstLayers;
	HiddenLayer *lastLayer;

#if CPU_MODE
protected:
	rcube delta_input;
#else
protected:
	const float alpha=1.0f, beta=0.0f;
	DATATYPE *d_delta_input;
#endif




};



#endif /* LAYER_INCEPTIONLAYER_H_ */
