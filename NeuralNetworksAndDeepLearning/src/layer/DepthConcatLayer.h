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
	DepthConcatLayer() { this->type = LayerType::DepthConcat; }
	DepthConcatLayer(const char *name);
	virtual ~DepthConcatLayer();

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

	virtual void shape(UINT idx, io_dim in_dim);
	virtual void reshape(UINT idx, io_dim in_dim);
	virtual void clearShape(UINT idx);

#if CPU_MODE
public:
	DepthConcatLayer(const char *name, int n_in);
	rcube &getDeltaInput();
	void feedforward(UINT idx, const rcube &input);
#else
	/**
	 * DepthConcatLayer의 getDetalInput()의 경우,
	 * 내부적으로 호출횟수를 카운트하므로 절대 로깅용으로 호출해서는 안됨.
	 */
	DATATYPE *getDeltaInput();
	void feedforward(UINT idx, const DATATYPE *input);
#endif


protected:
	void initialize();
	virtual void _shape();
	virtual void _clearShape();


	int offsetIndex;

#if CPU_MODE
protected:
	rcube delta_input;
	rcube delta_input_sub;

	vector<int> offsets;
#else
protected:
	const float alpha=1.0f, beta=0.0f;
	DATATYPE *d_delta_input;
#endif



};



#endif /* LAYER_DEPTHCONCATLAYER_H_ */
