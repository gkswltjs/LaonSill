/**
 * @file	FullyConnectedLayer.h
 * @date	2016/5/10
 * @author	jhkim
 * @brief
 * @details
 */

#ifndef LAYER_FULLYCONNECTEDLAYER_H_
#define LAYER_FULLYCONNECTEDLAYER_H_

#include "HiddenLayer.h"
#include "LearnableLayer.h"
#include "LayerConfig.h"
#include "../activation/Activation.h"
#include "../activation/ActivationFactory.h"
#include "../cost/Cost.h"





/**
 * @brief Fully Connected (Inner Product) 레이어
 * @details 이전 레이어와 현재 레이어의 모든 노드들에 대해 연결성이 있고
 *          연결성을 통해 weighted sum, activation을 수행 출력값을 계산하는 레이어이다.
 *          입력 레이어가 다차원인 경우(이미지의 경우 height x width x channel의 3차원) 1차원으로 flatten((height*width*channel) x 1 x 1)된다.
 *          출력 역시 1차원 flatten 결과이며 필요에 따라서 입력받는 레이어에서 다시 차원을 복구해야 한다.
 */
class FullyConnectedLayer : public HiddenLayer, public LearnableLayer {
public:
	class Builder : public HiddenLayer::Builder {
	public:
		uint32_t _nOut;
		double _pDropout;
		update_param _weightUpdateParam;
		update_param _biasUpdateParam;
		param_filler _weightFiller;
		param_filler _biasFiller;
		Activation::Type _activationType;

		Builder() {
			_nOut = 0;
			_pDropout = 0.0;
			_weightUpdateParam.lr_mult = 1.0;
			_weightUpdateParam.decay_mult = 0.0;
			_biasUpdateParam.lr_mult = 1.0;
			_biasUpdateParam.decay_mult = 0.0;
			_weightFiller.type = ParamFillerType::Constant;
			_biasFiller.type = ParamFillerType::Constant;
			_activationType = Activation::NoActivation;
		}
		Builder* nOut(uint32_t nOut) {
			this->_nOut = nOut;
			return this;
		}
		Builder* pDropout(uint32_t pDropout) {
			this->_pDropout = pDropout;
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
		Builder* weightFiller(ParamFillerType weightFillerType, double value) {
			this->_weightFiller.type = weightFillerType;
			this->_weightFiller.value = value;
			return this;
		}
		Builder* biasFiller(ParamFillerType paramFillerType, double value) {
			this->_biasFiller.type = paramFillerType;
			this->_biasFiller.value = value;
			return this;
		}
		Builder* activationType(Activation::Type activationType) {
			this->_activationType = activationType;
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
			return new FullyConnectedLayer(this);
		}
	};

	/**
	 * @details FullyConnectedLayer 기본 생성자
	 *          내부적으로 레이어 타입만 초기화한다.
	 */
	FullyConnectedLayer();
	FullyConnectedLayer(Builder* builder);
	/**
	 * @details FullyConnectedLayer 생성자
	 * @param name 레이어의 이름 문자열 포인터
	 * @param n_out 출력 노드의 수
	 * @param p_dropout dropout을 적용할 확율
	 * @param weight_update_param weight 갱신 관련 파라미터 구조체
	 * @param bias_update_param bias 갱신 관련 파라미터 구조체
	 * @param weight_filler weight 초기화 관련 파라미터 구조체
	 * @param bias_filler bias 초기화 관련 파라미터 구조체
	 * @param activationType weighted sum에 적용할 활성화 타입
	 */
	FullyConnectedLayer(const string name, int n_out, double p_dropout, update_param weight_update_param, update_param bias_update_param,
			param_filler weight_filler, param_filler bias_filler, Activation::Type activationType=Activation::NoActivation);
#ifndef GPU_MODE
	FullyConnectedLayer(const string name, int n_in, int n_out, double p_dropout, update_param weight_update_param, update_param bias_update_param,
			param_filler weight_filler, param_filler bias_filler, Activation::Type activationType=Activation::NoActivation);
#endif
	virtual ~FullyConnectedLayer();









private:
	void initialize(int n_out, double p_dropout, update_param weight_update_param, update_param bias_update_param,
			param_filler weight_filler, param_filler bias_filler, Activation::Type activationType);

protected:
	virtual void _shape(bool recursive=true);
	virtual void _clearShape();
	//virtual double _sumSquareGrad();
	//virtual double _sumSquareParam();
	virtual DATATYPE sumSquareParamsData();
	virtual DATATYPE sumSquareParamsGrad();
	virtual void scaleParamsGrad(DATATYPE scale);
	virtual void _save(ofstream &ofs);
	virtual void _load(ifstream &ifs, map<Layer *, Layer *> &layerMap);
	//virtual void _update(UINT n, UINT miniBatchSize);
	virtual void update();
#ifndef GPU_MODE
	/**
	 * 주어진 입력 input에 대해 출력 activation을 계산
	 * @param input: 레이어 입력 데이터 (이전 레이어의 activation)
	 */
	virtual void _feedforward();
	virtual void reset_nabla(UINT idx);
#else
	virtual void _feedforward();
#endif
	virtual void _backpropagation();

	enum ParamType {
		Weight = 0,
		Bias = 1
	};





protected:
	double p_dropout;						///< dropout을 적용할 확율

	update_param weight_update_param;		///< weight 갱신 관련 파라미터 구조체
	update_param bias_update_param;			///< bias 갱신 관련 파라미터 구조체

	param_filler weight_filler;				///< weight 초기화 관련 파라미터 구조체
	param_filler bias_filler;				///< bias 초기화 관련 파라미터 구조체

	Activation *activation_fn;				///< 활성화 객체

#ifndef GPU_MODE
	rmat weight;
	rvec bias;

	rvec nabla_b;
	rmat nabla_w;

	rcube z;
	rcube delta;
	rcube delta_input;
#else
	//DATATYPE *weight;						///< weight 호스트 메모리 포인터 (초기화 및 읽기, 쓰기용)
	//DATATYPE *bias;						///< bias 호스트 메모리 포인터 (초기화 및 읽기, 쓰기용)

	//DATATYPE *d_weight;					///< weight 장치 메모리 포인터
	//DATATYPE *d_bias;						///< bias 장치 메모리 포인터

	//DATATYPE *d_z;							///< weighted sum 장치 메모리 포인터
	//DATATYPE *d_delta;						///< 네트워크 cost의 z(weighted sum)에 관한 gradient 장치 메모리 포인터
	//DATATYPE *d_delta_weight;				///< 네트워크 cost의 weight에 관한 gradient 장치 메모리 포인터
	//DATATYPE *d_delta_weight_prev;		///< 이전 업데이트의 네트워크 cost의 weight에 관한 gradient 장치 메모리 포인터 (momentum 계산용)
	//DATATYPE *d_delta_bias;				///< 네트워크 cost의 bias에 관한 gradient 장치 메모리 포인터
	//DATATYPE *d_delta_bias_prev;			///< 이전 업데이트의 네트워크 cost의 bias에 관한 gradient 장치 메모리 포인터 (momentum 계산용)

	Data* _preActivation;
	vector<Data*> _params;
	vector<Data*> _paramsHistory;

	DATATYPE *d_onevec;						///< batch 사이즈의 1 벡터, bias를 weighted sum에 더해 줄 때 사용

	DATATYPE *mask;							///< dropout 마스크 호스트 메모리 포인터
	DATATYPE *d_mask;						///< dropout 마스크 장치 메모리 포인터
	DATATYPE scale;							///< dropout 스케일 팩터
#endif

};






#endif /* LAYER_FULLYCONNECTEDLAYER_H_ */
