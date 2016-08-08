/**
 * @file	InceptionLayer.h
 * @date	2016/5/27
 * @author	jhkim
 * @brief
 * @details
 */

#ifndef LAYER_INCEPTIONLAYER_H_
#define LAYER_INCEPTIONLAYER_H_

#include "InputLayer.h"
#include "HiddenLayer.h"


/**
 * @brief GoogLeNet의 Inception Module을 구현한 레이어
 * @details GoogLeNet의 Inception Module with Dimensionality Reduction을 구현,
 *          Caffe의 경우 매 Inception Module마다 8개의 레이어를 직접 올렸으나, 이를 하나의 레이어로 추상화,
 * @todo 자체 레이어 연결성을 포함한 특수한 레이어라서 레이어 기능을 구현할 때마다 특수 기능을 구현해야 해서 번거로움,
 *       삭제하고 Caffe처럼 가는 것이 적합할 수 있음.
 */
class InceptionLayer : public HiddenLayer {
public:
	InceptionLayer() { this->type = LayerType::Inception; }
	/**
	 * @details InceptionLayer 생성자
	 * @param name 레이어 이름 문자열 포인터
	 * @param ic 인셉션 레이어 입력의 채널 수
	 * @param oc_cv1x1 1x1 컨볼루션 레이어의 출력 채널 수
	 * @param oc_cv3x3reduce 3x3 리덕션 컨볼루션 레이어의 출력 채널 수
	 * @param oc_cv3x3 3x3 컨볼루션 레이어의 출력 채널 수
	 * @param oc_cv5x5reduce 5x5 리덕션 컨볼루션 레이어의 출력 채널 수
	 * @param oc_cv5x5 5x5 컨볼루션 레이어의 출력 채널 수
	 * @param oc_cp 프로젝션 컨볼루션 레이어의 출력 채널 수
	 * @param weight_update_param weight 갱신 관련 파라미터 구조체
	 * @param bias_update_param bias 갱신 관련 파라미터 구조체
	 */
	InceptionLayer(const char *name, int ic, int oc_cv1x1, int oc_cv3x3reduce, int oc_cv3x3, int oc_cv5x5reduce, int oc_cv5x5, int oc_cp,
			update_param weight_update_param, update_param bias_update_param);
	virtual ~InceptionLayer();

	virtual DATATYPE *getOutput() { return lastLayer->getOutput(); }

	void backpropagation(UINT idx, DATATYPE *next_delta_input);
	void reset_nabla(UINT idx);
	void update(UINT idx, UINT n, UINT miniBatchSize);

	void load(ifstream &ifs, map<Layer *, Layer *> &layerMap);
	/**
	 * @details 인센셥 레이어의 내부 네트워크의 메타 정보를 쓴다.
	 * @param idx 현재 레이어에 연결된 이전 레이어의 순번 index
	 * @param ofs 레이어를 쓸 출력 스트림
	 */
	void saveNinHeader(UINT idx, ofstream &ofs);
	virtual Layer* find(UINT idx, const char* name);

	virtual bool isLearnable() { return true; }


#if CPU_MODE
public:
	InceptionLayer(const char *name, int n_in, int n_out, int cv1x1, int cv3x3reduce, int cv3x3, int cv5x5reduce, int cv5x5, int cp);
	rcube &getDeltaInput() { return this->delta_input; }
	void feedforward(UINT idx, const rcube &input, const char *end=0);
#else
public:
	DATATYPE *getDeltaInput() { return this->d_delta_input; }
	void feedforward(UINT idx, const DATATYPE *input, const char *end=0);
#endif

protected:
	void initialize();
	void initialize(int ic, int cv1x1, int cv3x3reduce, int cv3x3, int cv5x5reduce, int cv5x5, int cp,
			update_param weight_update_param, update_param bias_update_param);

	virtual void _save(ofstream &ofs);
	virtual void _shape(bool recursive=true);
	virtual void _reshape();
	virtual void _clearShape();
	virtual DATATYPE _sumSquareParam();
	virtual DATATYPE _sumSquareParam2();
	virtual void _scaleParam(DATATYPE scale_factor);

	//InputLayer *inputLayer;
	vector<HiddenLayer *> firstLayers;				///< 인셉션 레이어 내부 네트워크의 시작 레이어 포인터 목록 벡터
	HiddenLayer *lastLayer;							///< 인셉션 레이어 내부 네트워크의 출력 레이어 포인터

#if CPU_MODE
protected:
	rcube delta_input;
#else
protected:
	const float alpha=1.0f, beta=0.0f;				///< cudnn 함수에서 사용하는 scaling factor, 다른 곳으로 옮겨야 함.




};



#endif /* LAYER_INCEPTIONLAYER_H_ */
