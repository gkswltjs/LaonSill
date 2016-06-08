/*
 * HiddenLayer.h
 *
 *  Created on: 2016. 5. 11.
 *      Author: jhkim
 */

#ifndef LAYER_HIDDENLAYER_H_
#define LAYER_HIDDENLAYER_H_

#include "Layer.h"
#include <armadillo>

using namespace arma;


class HiddenLayer : public Layer {
public:
	HiddenLayer(string name, int n_in, int n_out) : Layer(name, n_in, n_out) {}
	HiddenLayer(string name, io_dim in_dim, io_dim out_dim) : Layer(name, in_dim, out_dim) {}
	virtual ~HiddenLayer() {}


	virtual rcube &getDeltaInput()=0;
	vector<prev_layer_relation> &getPrevLayers() { return this->prevLayers; }
	int getPrevLayerSize() { return this->prevLayers.size(); }



	/**
	 * 네트워크 cost에 대한 weight update양 계산
	 * @param next_w: 다음 레이어의 weight
	 * @param input: 레이어 입력 데이터 (이전 레이어의 activation)
	 */
	//virtual void backpropagation(HiddenLayer *next_layer)=0;
	virtual void backpropagation(UINT idx, HiddenLayer *next_layer) {
		for(UINT i = 0; i < prevLayers.size(); i++) {
			prevLayers[i].prev_layer->backpropagation(prevLayers[i].idx, this);
		}
	}

	/**
	 * 한 번의 batch 종료 후 재사용을 위해 w, b 누적 업데이트를 reset
	 */
	virtual void reset_nabla(UINT idx)=0;

	/**
	 * 한 번의 batch 종료 후 w, b 누적 업데이트를 레이어 w, b에 적용
	 * @param eta:
	 * @param lambda:
	 * @param n:
	 * @param miniBatchSize:
	 */
	virtual void update(UINT idx, int n, int miniBatchSize)=0;

	void addPrevLayer(prev_layer_relation prevLayer) { prevLayers.push_back(prevLayer); }







protected:
	bool isLastNextLayerRequest(UINT idx) {
		//cout << name << " received request from " << idx << "th next layer ... " << endl;
		if(nextLayers.size() > idx+1) {
			//cout << name << " is not from last next layer... " << endl;
			return false;
		} else {
			return true;
		}
	}


	vector<bool> backwardFlags;

};

#endif /* LAYER_HIDDENLAYER_H_ */



















