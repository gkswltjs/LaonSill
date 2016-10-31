/**
 * @file SoftmaxLayer.h
 * @date 2016/8/1
 * @author jhkim
 * @brief
 * @details
 */

#ifndef SOFTMAXLAYER_H_
#define SOFTMAXLAYER_H_

#include "../common.h"
#include "OutputLayer.h"
#include "../cost/LogLikelihoodCost.h"
#include "../activation/Softmax.h"
#include "../exception/Exception.h"




/**
 * @brief 소프트맥스 출력 레이어
 * @details 활성화 함수로 Softmax를, cost 함수로 LogLikelihood를 적용시킨 출력 레이어
 */
template <typename Dtype>
class SoftmaxLayer : public OutputLayer<Dtype> {
public:
	/**
	 * @brief 소프트맥스 출력 레이어 객체 빌더
	 * @details 소프트맥스 출력 레이어를 생성할 때 필요한 파라미터들을 설정하고 build()를 통해
	 *          해당 파라미터를 만족하는 소프트 맥스 레이어 객체를 생성한다.
	 */
	class Builder : public OutputLayer<Dtype>::Builder {
	public:
		Builder() {
			this->type = Layer<Dtype>::Softmax;
			this->_activationType = Activation<Dtype>::Softmax;
			this->_costType = Cost<Dtype>::LogLikelihood;
		}
		Builder* costType(typename Cost<Dtype>::Type costType) {
			OutputLayer<Dtype>::Builder::costType(costType);
			return this;
		}
		Builder* nOut(uint32_t nOut) {
			OutputLayer<Dtype>::Builder::nOut(nOut);
			return this;
		}
		Builder* pDropout(double pDropout) {
			OutputLayer<Dtype>::Builder::pDropout(pDropout);
			return this;
		}
		Builder* weightUpdateParam(double lr_mult, double decay_mult) {
			OutputLayer<Dtype>::Builder::weightUpdateParam(lr_mult, decay_mult);
			return this;
		}
		Builder* biasUpdateParam(double lr_mult, double decay_mult) {
			OutputLayer<Dtype>::Builder::biasUpdateParam(lr_mult, decay_mult);
			return this;
		}
		Builder* weightFiller(ParamFillerType weightFillerType, double value) {
			OutputLayer<Dtype>::Builder::weightFiller(weightFillerType, value);
			return this;
		}
		Builder* biasFiller(ParamFillerType biasFillerType, double value) {
			OutputLayer<Dtype>::Builder::biasFiller(biasFillerType, value);
			return this;
		}
		Builder* activationType(typename Activation<Dtype>::Type activationType) {
			OutputLayer<Dtype>::Builder::activationType(activationType);
			return this;
		}
		virtual Builder* name(const std::string name) {
			OutputLayer<Dtype>::Builder::name(name);
			return this;
		}
		virtual Builder* id(uint32_t id) {
			OutputLayer<Dtype>::Builder::id(id);
			return this;
		}
		virtual Builder* nextLayerIndices(const std::vector<uint32_t>& nextLayerIndices) {
			OutputLayer<Dtype>::Builder::nextLayerIndices(nextLayerIndices);
			return this;
		}
		virtual Builder* prevLayerIndices(const std::vector<uint32_t>& prevLayerIndices) {
			OutputLayer<Dtype>::Builder::prevLayerIndices(prevLayerIndices);
			return this;
		}
		Layer<Dtype>* build() {
			return new SoftmaxLayer(this);
		}
		virtual void save(std::ofstream& ofs) {
			OutputLayer<Dtype>::Builder::save(ofs);
		}
		virtual void load(std::ifstream& ifs) {
			OutputLayer<Dtype>::Builder::load(ifs);
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
	SoftmaxLayer(const std::string name, int n_out, double p_dropout, update_param weight_update_param, update_param bias_update_param,
			param_filler<Dtype> weight_filler, param_filler<Dtype> bias_filler);
	virtual ~SoftmaxLayer();

	/**
	 * @details 히든 레이어의 backpropagation()을 override
	 *          히든 레이어와 달리 출력 레이어는 Cost와 target값을 통해 gradient가 계산된다.
	 * @param target 현재 입력 데이터에 대한 정답 레이블
	 */
	using OutputLayer<Dtype>::backpropagation;
	//void backpropagation(const uint32_t* target);
	void backpropagation(DataSet<Dtype>* dataSet, const uint32_t baseIndex);

	/**
	 * @details 출력 레이어의 출력값과 데이터에 대한 정답으로 cost를 계산한다.
	 * @param target 데이터에 대한 정답 장치 메모리 포인터
	 */
	//double cost(const uint32_t *target);
	double cost(DataSet<Dtype>* dataSet, const uint32_t baseIndex);

protected:
	void initialize();

protected:
	virtual void _shape(bool recursive=true);
	virtual void _clearShape();
	//void _load(ifstream &ifs, map<Layer<Dtype>*, Layer<Dtype>*>& layerMap);




};



#endif /* SOFTMAXLAYER_H_ */
