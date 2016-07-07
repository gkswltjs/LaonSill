/*
 * Layer.h
 *
 *  Created on: 2016. 5. 10.
 *      Author: jhkim
 */

#ifndef LAYER_LAYER_H_
#define LAYER_LAYER_H_

#include "LayerConfig.h"
#include "../cuda/Cuda.h"
#include "../Util.h"
#include <armadillo>
#include <iostream>
#include <map>

using namespace arma;

const int LAYER_NAME_LENGTH = 32;

enum class LayerType {
	Input, FullyConnected, Conv, Pooling, DepthConcat, Inception, LRN, Sigmoid, Softmax
};


class Layer {

public:
	Layer() {}
	Layer(const char *name, io_dim in_dim, io_dim out_dim);
	virtual ~Layer();

	int getId() const { return id; }
	LayerType getType() { return this->type; }

	vector<next_layer_relation> &getNextLayers() { return this->nextLayers; }
	int getNextLayerSize() { return this->nextLayers.size(); }
	vector<prev_layer_relation> &getPrevLayers() { return this->prevLayers; }
	int getPrevLayerSize() { return this->prevLayers.size(); }
	io_dim getInDimension() const { return in_dim; }
	io_dim getOutDimension() const { return out_dim; }


	void addPrevLayer(prev_layer_relation prevLayer);
	void addNextLayer(next_layer_relation nextLayer);

	virtual void reset_nabla(UINT idx);
	virtual void update(UINT idx, UINT n, UINT miniBatchSize);

	virtual void save(UINT idx, ofstream &ofs);
	virtual void saveHeader(UINT idx, ofstream &ofs);
	virtual void load(ifstream &ifs, map<Layer *, Layer *> &layerMap);

	bool isLastPrevLayerRequest(UINT idx);
	bool isLastNextLayerRequest(UINT idx);


protected:
	void initialize(const char *name, io_dim in_dim, io_dim out_dim);
	void propResetNParam();
	void propUpdate(UINT n, UINT miniBatchSize);
	void propSave(ofstream &ofs);

	virtual void save(ofstream &ofs);
	virtual void loadNetwork(ifstream &ifs, map<Layer *, Layer *> &layerMap);
	virtual void updateLayerRelation(map<Layer *, Layer *> &layerMap);

	static int generateLayerId();

	LayerType type;
	int id;
	char name[32];

	io_dim in_dim;
	io_dim out_dim;

	vector<prev_layer_relation> prevLayers;
	vector<next_layer_relation> nextLayers;

	static int layerCount;

#if CPU_MODE
public:
	Layer(const char *name, int n_in, int n_out);

	rcube &getInput() { return this->input; }
	rcube &getOutput() { return this->output; }

	/**
	 * 주어진 입력 input에 대해 출력 activation을 계산
	 * @param input: 레이어 입력 데이터 (이전 레이어의 activation)
	 */
	// sub class에서 구현이 없을 때에만 참조, 구현이 있을 경우 prop*() 함수를 참조
	virtual void feedforward(UINT idx, const rcube &input);

protected:
	void propFeedforward(const rcube output);

	rcube input;
	rcube output;

#else

public:
	const DATATYPE *getInput() { return this->d_input; }
	DATATYPE *getOutput() { return this->d_output; }

	virtual void feedforward(UINT idx, const DATATYPE *input);

protected:
	void propFeedforward(const DATATYPE *output);

	//DATATYPE *input;
	//DATATYPE *output;

	const DATATYPE *d_input;		// input pointer is assigned from prev layer output pointer
	DATATYPE *d_output;					// has own device memory allocated

	cudnnTensorDescriptor_t inputTensorDesc;
	cudnnTensorDescriptor_t outputTensorDesc;

#endif



};






#endif /* LAYER_LAYER_H_ */































