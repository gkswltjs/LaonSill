/**
 * @file	InceptionLayer.h
 * @date	2016/5/27
 * @author	jhkim
 * @brief
 * @details
 */

#ifndef LAYER_INCEPTIONLAYER_H_
#define LAYER_INCEPTIONLAYER_H_


#include "../common.h"

#ifndef GPU_MODE
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
	class Builder : public HiddenLayer::Builder {
	public:
		uint32_t _ic;
		uint32_t _oc_cv1x1;
		uint32_t _oc_cv3x3reduce;
		uint32_t _oc_cv3x3;
		uint32_t _oc_cv5x5reduce;
		uint32_t _oc_cv5x5;
		uint32_t _oc_cp;
		update_param _weightUpdateParam;
		update_param _biasUpdateParam;

		Builder() {
			_ic = 0;
			_oc_cv1x1 = 0;
			_oc_cv3x3reduce = 0;
			_oc_cv3x3 = 0;
			_oc_cv5x5reduce = 0;
			_oc_cv5x5 = 0;
			_oc_cp = 0;
			_weightUpdateParam.lr_mult = 1.0;
			_weightUpdateParam.decay_mult = 0.0;
			_biasUpdateParam.lr_mult = 1.0;
			_biasUpdateParam.decay_mult = 0.0;
		}
		Builder* ic(uint32_t ic) {
			this->_ic = ic;
			return this;
		}
		Builder* oc_cv1x1(uint32_t oc_cv1x1) {
			this->_oc_cv1x1 = oc_cv1x1;
			return this;
		}
		Builder* oc_cv3x3reduce(uint32_t oc_cv3x3reduce) {
			this->_oc_cv3x3reduce = oc_cv3x3reduce;
			return this;
		}
		Builder* oc_cv3x3(uint32_t oc_cv3x3) {
			this->_oc_cv3x3 = oc_cv3x3;
			return this;
		}
		Builder* oc_cv5x5reduce(uint32_t oc_cv5x5reduce) {
			this->_oc_cv5x5reduce = oc_cv5x5reduce;
			return this;
		}
		Builder* oc_cv5x5(uint32_t oc_cv5x5) {
			this->_oc_cv5x5 = oc_cv5x5;
			return this;
		}
		Builder* oc_cp(uint32_t oc_cp) {
			this->_oc_cp = oc_cp;
			return this;
		}
		Builder* weightUpdateParam(double lr_mult, double decay_mult) {
			this->_weightUpdateParam.lr_mult = lr_mult;
			this->_weightUpdateParam.decay_mult = decay_mult;
			return this;
		}
		Builder* biasUpdateParam(double lr_mult, double decay_mult) {
			this->_biasUpdateParam.lr_mult = lr_mult;
			this->_biasUpdateParam.decay_mult = decay_mult;
			return this;
		}
		virtual Builder* name(const string name) {
			HiddenLayer::Builder::name(name);
			return this;
		}
		virtual Builder* id(uint32_t id) {
			HiddenLayer::Builder::id(id);
			return this;
		}
		virtual Builder* nextLayerIndices(const vector<uint32_t>& nextLayerIndices) {
			HiddenLayer::Builder::nextLayerIndices(nextLayerIndices);
			return this;
		}
		virtual Builder* prevLayerIndices(const vector<uint32_t>& prevLayerIndices) {
			HiddenLayer::Builder::prevLayerIndices(prevLayerIndices);
			return this;
		}
		Layer* build() {
			return new InceptionLayer(this);
		}
	};
	/**
	 * @details InceptionLayer 기본 생성자
	 */
	InceptionLayer();
	InceptionLayer(Builder* builder);
	/**
	 * @details InceptionLayer 생성자
	 * @param name 레이어 이름 문자열
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
	InceptionLayer(const string name, int ic, int oc_cv1x1, int oc_cv3x3reduce, int oc_cv3x3, int oc_cv5x5reduce, int oc_cv5x5, int oc_cp,
			update_param weight_update_param, update_param bias_update_param);
#ifndef GPU_MODE
	InceptionLayer(const string name, int n_in, int n_out, int cv1x1, int cv3x3reduce, int cv3x3, int cv5x5reduce, int cv5x5, int cp);
#endif
	/**
	 * @details InceptionLayer 소멸자
	 */
	virtual ~InceptionLayer();

	virtual DATATYPE *getOutput() { return lastLayer->getOutput(); }

	virtual void setNetworkConfig(NetworkConfig* networkConfig);

	/**
	 * @details 내부 레이어들에 'end' param을 전달해야 해서 재정의
	 *          _feedforward()만 재정의해서 'end'를 전달할 방법이 없다.
	 */
	virtual void feedforward(UINT idx, const DATATYPE* input, const char* end);

protected:
	void initialize();
	void initialize(int ic, int cv1x1, int cv3x3reduce, int cv3x3, int cv5x5reduce, int cv5x5, int cp,
			update_param weight_update_param, update_param bias_update_param);

	virtual Layer* _find(const string name);
	virtual void _shape(bool recursive=true);
	virtual void _reshape();
	virtual void _clearShape();
	virtual double _sumSquareGrad();
	virtual double _sumSquareParam();
	virtual void _save(ofstream &ofs);
	/**
	 * @details 인센셥 레이어의 내부 네트워크의 메타 정보를 쓴다.
	 * @param idx 현재 레이어에 연결된 이전 레이어의 순번 index
	 * @param ofs 레이어를 쓸 출력 스트림
	 */
	void _saveNinHeader(ofstream &ofs);
	void _load(ifstream &ifs, map<Layer *, Layer *> &layerMap);
	virtual void _update(UINT n, UINT miniBatchSize);
	/**
	 * @details 레이어로 전달된 gradient를 내부 네트워크로 전달하고,
	 *          내부 네트워크에서 backpropagation된 결과를 하나로 취합하여 레이어 입력에 관한 gradient로 구한다.
	 * @param next_delta_input 네트워크 cost의 다음 레이어의 입력에 관한 gradient 장치 메모리 포인터
	 */
	virtual void _backpropagation();

#ifndef GPU_MODE
	virtual void reset_nabla(UINT idx);
#else
	virtual void _scaleParam(DATATYPE scale_factor);
#endif


protected:
	//InputLayer *inputLayer;
	vector<HiddenLayer*> firstLayers;				///< 인셉션 레이어 내부 네트워크의 시작 레이어 포인터 목록 벡터
	HiddenLayer *lastLayer;							///< 인셉션 레이어 내부 네트워크의 출력 레이어 포인터
	vector<Layer*> layers;

#ifndef GPU_MODE
	rcube delta_input;
#else
#endif



};

#endif

#endif /* LAYER_INCEPTIONLAYER_H_ */
