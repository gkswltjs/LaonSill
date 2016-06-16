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



#if CPU_MODE


class HiddenLayer : public Layer {
public:
	HiddenLayer() {}
	HiddenLayer(const char *name, int n_in, int n_out) : Layer(name, n_in, n_out) {}
	HiddenLayer(const char *name, io_dim in_dim, io_dim out_dim) : Layer(name, in_dim, out_dim) {}
	virtual ~HiddenLayer() {}

	virtual rcube &getDeltaInput()=0;

	/**
	 * 네트워크 cost에 대한 weight update양 계산
	 * @param next_w: 다음 레이어의 weight
	 * @param input: 레이어 입력 데이터 (이전 레이어의 activation)
	 */
	//virtual void backpropagation(HiddenLayer *next_layer)=0;
	virtual void backpropagation(UINT idx, HiddenLayer *next_layer) { propBackpropagation(); }

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
	virtual void update(UINT idx, UINT n, UINT miniBatchSize)=0;





	//virtual void save(UINT idx, ofstream &ofs) {
	//	save(ofs);
	//	propSave(ofs);
	//}

	//virtual void load(ifstream &ifs, map<Layer *, Layer *> &layerMap) {
	//	Layer::load(ifs, layerMap);
	//}





protected:

	void propBackpropagation() {
		HiddenLayer *hiddenLayer;
		for(UINT i = 0; i < prevLayers.size(); i++) {
			hiddenLayer = dynamic_cast<HiddenLayer *>(prevLayers[i].prev_layer);
			if(hiddenLayer) hiddenLayer->backpropagation(prevLayers[i].idx, this);
		}
	}

	//virtual void save(ofstream &ofs) {
	//	Layer::save(ofs);
	//}


};


#else


class HiddenLayer : public Layer {
public:
	HiddenLayer() {}
	HiddenLayer(const char *name, int n_in, int n_out) : Layer(name, n_in, n_out) {}
	HiddenLayer(const char *name, io_dim in_dim, io_dim out_dim) : Layer(name, in_dim, out_dim) {}
	virtual ~HiddenLayer() {}

	virtual rcube &getDeltaInput()=0;

	/**
	 * 네트워크 cost에 대한 weight update양 계산
	 * @param next_w: 다음 레이어의 weight
	 * @param input: 레이어 입력 데이터 (이전 레이어의 activation)
	 */
	//virtual void backpropagation(HiddenLayer *next_layer)=0;
	virtual void backpropagation(UINT idx, HiddenLayer *next_layer) { propBackpropagation(); }

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
	virtual void update(UINT idx, UINT n, UINT miniBatchSize)=0;





	//virtual void save(UINT idx, ofstream &ofs) {
	//	save(ofs);
	//	propSave(ofs);
	//}

	//virtual void load(ifstream &ifs, map<Layer *, Layer *> &layerMap) {
	//	Layer::load(ifs, layerMap);
	//}





protected:

	void propBackpropagation() {
		HiddenLayer *hiddenLayer;
		for(UINT i = 0; i < prevLayers.size(); i++) {
			hiddenLayer = dynamic_cast<HiddenLayer *>(prevLayers[i].prev_layer);
			if(hiddenLayer) hiddenLayer->backpropagation(prevLayers[i].idx, this);
		}
	}

	//virtual void save(ofstream &ofs) {
	//	Layer::save(ofs);
	//}


};


#endif





#endif /* LAYER_HIDDENLAYER_H_ */



















