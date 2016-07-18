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
	LRNLayer() { this->type = LayerType::LRN; }
	LRNLayer(const char *name, lrn_dim lrn_d);
	virtual ~LRNLayer();

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

	void load(ifstream &ifs, map<Layer *, Layer *> &layerMap);

#if CPU_MODE
public:
	rcube &getDeltaInput() { return delta_input; }
	void feedforward(UINT idx, const rcube &input, const char *end=0);
#else
public:
	DATATYPE *getDeltaInput() { return d_delta_input; }
	void feedforward(UINT idx, const DATATYPE *input, const char *end=0);
#endif


protected:
	void initialize(lrn_dim lrn_d);

	virtual void _save(ofstream &ofs);
	virtual void _shape(bool recursive=true);
	virtual void _clearShape();

	lrn_dim lrn_d;

#if CPU_MODE
private:
	rcube delta_input;
	rcube z;	// beta powered 전의 weighted sum 상태의 norm term
#else
private:
	const float alpha=1.0f, beta=0.0f;

	//DATATYPE *d_delta;
	DATATYPE *d_delta_input;

	cudnnLRNDescriptor_t lrnDesc;
#endif

};




#endif /* LAYER_LRNLAYER_H_ */
