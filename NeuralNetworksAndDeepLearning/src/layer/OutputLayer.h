/**
 * @file OutputLayer.h
 * @date 2016/5/12
 * @author jhkim
 * @brief
 * @details
 */


#ifndef LAYER_OUTPUTLAYER_H_
#define LAYER_OUTPUTLAYER_H_

#include "common.h"
#include "FullyConnectedLayer.h"
#include "CostFactory.h"
#include "SyncMem.h"
#include "DataSet.h"


/**
 * @brief 출력 레이어 기본 추상 클래스
 * @details FullyConnectedLayer를 상속, 출력 레이어는 항상 FullyConnectedLayer만이 될 수 있다.
 *          cost() method를 구현한다.
 */
template <typename Dtype>
class OutputLayer : public FullyConnectedLayer<Dtype> {
public:
	/**
	 * @brief 출력 레이어 객체 빌더
	 * @details 출력 레이어를 생성할 때 필요한 파라미터들을 설정하고 build()를 통해
	 *          해당 파라미터를 만족하는 출력 레이어 객체를 생성한다.
	 */
	class Builder : public FullyConnectedLayer<Dtype>::Builder {
	public:
		typename Cost<Dtype>::Type _costType;

		Builder() {
			_costType = Cost<Dtype>::NoCost;
		}
		Builder* costType(typename Cost<Dtype>::Type costType) {
			this->_costType = costType;
			return this;
		}
		Builder* nOut(uint32_t nOut) {
			FullyConnectedLayer<Dtype>::Builder::nOut(nOut);
			return this;
		}
		Builder* pDropout(double pDropout) {
			FullyConnectedLayer<Dtype>::Builder::pDropout(pDropout);
			return this;
		}
		Builder* weightUpdateParam(double lr_mult, double decay_mult) {
			FullyConnectedLayer<Dtype>::Builder::weightUpdateParam(lr_mult, decay_mult);
			return this;
		}
		Builder* biasUpdateParam(double lr_mult, double decay_mult) {
			FullyConnectedLayer<Dtype>::Builder::biasUpdateParam(lr_mult, decay_mult);
			return this;
		}
		Builder* weightFiller(ParamFillerType weightFillerType, double value) {
			FullyConnectedLayer<Dtype>::Builder::weightFiller(weightFillerType, value);
			return this;
		}
		Builder* biasFiller(ParamFillerType biasFillerType, double value) {
			FullyConnectedLayer<Dtype>::Builder::biasFiller(biasFillerType, value);
			return this;
		}
		Builder* activationType(typename Activation<Dtype>::Type activationType) {
			FullyConnectedLayer<Dtype>::Builder::activationType(activationType);
			return this;
		}
		virtual Builder* name(const std::string name) {
			FullyConnectedLayer<Dtype>::Builder::name(name);
			return this;
		}
		virtual Builder* id(uint32_t id) {
			FullyConnectedLayer<Dtype>::Builder::id(id);
			return this;
		}
		virtual Builder* nextLayerIndices(const std::vector<uint32_t>& nextLayerIndices) {
			FullyConnectedLayer<Dtype>::Builder::nextLayerIndices(nextLayerIndices);
			return this;
		}
		virtual Builder* prevLayerIndices(const std::vector<uint32_t>& prevLayerIndices) {
			FullyConnectedLayer<Dtype>::Builder::prevLayerIndices(prevLayerIndices);
			return this;
		}
		Layer<Dtype>* build() = 0;
		virtual void save(std::ofstream& ofs) {
			FullyConnectedLayer<Dtype>::Builder::save(ofs);
			ofs.write((char*)&_costType, sizeof(typename Cost<Dtype>::Type));
		}
		virtual void load(std::ifstream& ifs) {
			FullyConnectedLayer<Dtype>::Builder::load(ifs);
			ifs.read((char*)&_costType, sizeof(typename Cost<Dtype>::Type));
		}
	};

	OutputLayer() {}
	OutputLayer(Builder* builder) : FullyConnectedLayer<Dtype>(builder) {
		initialize(builder->_costType);
	}
	/**
	 * @details OutputLayer 생성자
	 * @param name 레이어의 이름 문자열 포인터
	 * @param n_out 출력 노드의 수
	 * @param p_dropout dropout을 적용할 확율
	 * @param weight_update_param weight 갱신 관련 파라미터 구조체
	 * @param bias_update_param bias 갱신 관련 파라미터 구조체
	 * @param weight_filler weight 초기화 관련 파라미터 구조체
	 * @param bias_filler bias 초기화 관련 파라미터 구조체
	 * @param activationType weighted sum에 적용할 활성화 타입
	 * @param costType 레이어 출력값에 대한 cost 계산 타입
	 */
	OutputLayer(const std::string name, int n_out, double p_dropout, update_param weight_update_param, update_param bias_update_param,
			param_filler<Dtype> weight_filler, param_filler<Dtype> bias_filler, typename Activation<Dtype>::Type activationType, typename Cost<Dtype>::Type costType)
		:FullyConnectedLayer<Dtype>(name, n_out, p_dropout, weight_update_param, bias_update_param, weight_filler, bias_filler, activationType) {
		initialize(costType);
	};
	virtual ~OutputLayer() {
		CostFactory<Dtype>::destroy(cost_fn);
	};

	/**
	 * @details 히든 레이어의 backpropagation()을 override
	 *          히든 레이어와 달리 출력 레이어는 Cost와 target값을 통해 gradient가 계산된다.
	 * @param target 현재 입력 데이터에 대한 정답 레이블
	 */
	using FullyConnectedLayer<Dtype>::backpropagation;
	//virtual void backpropagation(const uint32_t* target) = 0;
	virtual void backpropagation(DataSet<Dtype>* dataSet, const uint32_t baseIndex) = 0;

	/**
	 * @details 출력레이어의 cost를 게산한다.
	 * @param target 현재 cost를 구한 데이터에 대한 정답 장치 메모리 포인터
	 * @todo softmax-loglikelihood의 경우 아주 단순한 형태여서
	 *       별도로 cost를 구하고 gradient를 다시 계산할 경우 효율적이지 못해서 cost에서 gradient까지 계산하게 되어있다.
	 *       하지만 적절한 modularity를 달성하기 위해서 cost를 구하는 것과 gradient를 계산하는 것은 구분되어야 한다.
	 */
	//virtual double cost(const uint32_t* target) = 0;
	virtual double cost(DataSet<Dtype>* dataSet, const uint32_t baseIndex) = 0;


protected:
	void initialize(typename Cost<Dtype>::Type costType) {
		//if(this->activation_fn) this->activation_fn->initialize_weight(in_dim.rows, weight);
		this->cost_fn = CostFactory<Dtype>::create(costType);
	}

	virtual void _shape(bool recursive=true) {
		if(recursive) {
			FullyConnectedLayer<Dtype>::_shape();
		}
		this->_target.shape(this->out_dim.batches);
	}
	virtual void _clearShape() {
		FullyConnectedLayer<Dtype>::_clearShape();
	}

	/*
	virtual void _save(std::ofstream &ofs) {
		FullyConnectedLayer<Dtype>::_save(ofs);
		//int costType = (int)cost_fn->getType();
		//ofs.write((char *)&costType, sizeof(int));
	}
	virtual void _load(std::ifstream &ifs, map<Layer<Dtype>*, Layer<Dtype>*>& layerMap) {
		FullyConnectedLayer<Dtype>::_load(ifs, layerMap);

		OutputLayer<Dtype>::_shape(false);
		//typename Cost<Dtype>::Type type;
		//ifs.read((char *)&type, sizeof(int));
		//initialize(type);
	}
	*/

protected:
	Cost<Dtype>* cost_fn;				///< cost 객체

	SyncMem<uint32_t> _target;
};



template class OutputLayer<float>;


#endif /* LAYER_OUTPUTLAYER_H_ */








































