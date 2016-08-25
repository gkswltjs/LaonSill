/**
 * @file SoftmaxLayer.h
 * @date 2016/8/1
 * @author jhkim
 * @brief
 * @details
 */

#ifndef SOFTMAXLAYER_H_
#define SOFTMAXLAYER_H_

#include "OutputLayer.h"
#include "../cost/LogLikelihoodCost.h"
#include "../activation/Softmax.h"
#include "../exception/Exception.h"
#ifndef GPU_MODE
#include <armadillo>
#endif

#ifndef GPU_MODE
using namespace arma;
#endif








/**
 * @brief 소프트맥스 출력 레이어
 * @details 활성화 함수로 Softmax를, cost 함수로 LogLikelihood를 적용시킨 출력 레이어
 */
class SoftmaxLayer : public OutputLayer {
public:
	class Builder : public OutputLayer::Builder {
	public:
		Builder() {
			_activationType = Activation::Softmax;
			_costType = Cost::LogLikelihood;
		}
		Builder* costType(Cost::Type costType) {
			OutputLayer::Builder::costType(costType);
			return this;
		}
		Builder* nOut(uint32_t nOut) {
			OutputLayer::Builder::nOut(nOut);
			return this;
		}
		Builder* pDropout(uint32_t pDropout) {
			OutputLayer::Builder::pDropout(pDropout);
			return this;
		}
		Builder* weightUpdateParam(double lr_mult, double decay_mult) {
			OutputLayer::Builder::weightUpdateParam(lr_mult, decay_mult);
			return this;
		}
		Builder* biasUpdateParam(double lr_mult, double decay_mult) {
			OutputLayer::Builder::biasUpdateParam(lr_mult, decay_mult);
			return this;
		}
		Builder* weightFiller(ParamFillerType weightFillerType, double value) {
			OutputLayer::Builder::weightFiller(weightFillerType, value);
			return this;
		}
		Builder* biasFiller(ParamFillerType biasFillerType, double value) {
			OutputLayer::Builder::biasFiller(biasFillerType, value);
			return this;
		}
		Builder* activationType(Activation::Type activationType) {
			OutputLayer::Builder::activationType(activationType);
			return this;
		}
		virtual Builder* name(const string name) {
			OutputLayer::Builder::name(name);
			return this;
		}
		virtual Builder* id(uint32_t id) {
			OutputLayer::Builder::id(id);
			return this;
		}
		virtual Builder* nextLayerIndices(const vector<uint32_t>& nextLayerIndices) {
			OutputLayer::Builder::nextLayerIndices(nextLayerIndices);
			return this;
		}
		virtual Builder* prevLayerIndices(const vector<uint32_t>& prevLayerIndices) {
			OutputLayer::Builder::prevLayerIndices(prevLayerIndices);
			return this;
		}
		Layer* build() {
			return new SoftmaxLayer(this);
		}
	};

	SoftmaxLayer();
	SoftmaxLayer(Builder* builder);
	/**
	 * @details SoftmaxLayer 생성자
	 * @param name 레이어의 이름 문자열 포인터
	 * @param n_out 출력 노드의 수
	 * @param p_dropout dropout을 적용할 확율
	 * @param weight_update_param weight 갱신 관련 파라미터 구조체
	 * @param bias_update_param bias 갱신 관련 파라미터 구조체
	 * @param weight_filler weight 초기화 관련 파라미터 구조체
	 * @param bias_filler bias 초기화 관련 파라미터 구조체
	 */
	SoftmaxLayer(const string name, int n_out, double p_dropout, update_param weight_update_param, update_param bias_update_param,
			param_filler weight_filler, param_filler bias_filler);
#ifndef GPU_MODE
	SoftmaxLayer(const string name, int n_in, int n_out, double p_dropout, update_param weight_update_param, update_param bias_update_param,
			param_filler weight_filler, param_filler bias_filler);
#endif
	virtual ~SoftmaxLayer();



#ifndef GPU_MODE
	void cost(const rvec &target);
#else
	/**
	 * @details 출력 레이어의 출력값과 데이터에 대한 정답으로 cost를 계산한다.
	 * @param target 데이터에 대한 정답 장치 메모리 포인터
	 */
	void cost(const uint32_t *target);
#endif


protected:
	void initialize();


protected:
	virtual void _shape(bool recursive=true);
	virtual void _clearShape();
	void _load(ifstream &ifs, map<Layer *, Layer *> &layerMap);




};



#endif /* SOFTMAXLAYER_H_ */
