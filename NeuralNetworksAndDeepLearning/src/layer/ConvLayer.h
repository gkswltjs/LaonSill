/**
 * @file	ConvLayer.h
 * @date	2016/5/23
 * @author	jhkim
 * @brief
 * @details
 */

#ifndef LAYER_CONVLAYER_H_
#define LAYER_CONVLAYER_H_

#include <stddef.h>
#include <iostream>
#include <map>
#include <string>

#include "common.h"
#include "Activation.h"
#include "Util.h"
#include "HiddenLayer.h"
#include "LearnableLayer.h"
#include "LayerConfig.h"

/**
 * @brief 컨볼루션 레이어
 * @details
 */
template <typename Dtype>
class ConvLayer : public HiddenLayer<Dtype>, public LearnableLayer<Dtype> {
public:
	/**
	 * @brief 컨볼루션 레이어 객체 빌더
	 * @details 컨볼루션 레이어를 생성할 때 필요한 파라미터들을 설정하고 build()를 통해
	 *          해당 파라미터를 만족하는 컨볼루션 레이어 객체를 생성한다.
	 */
	class Builder : public HiddenLayer<Dtype>::Builder {
	public:
		filter_dim _filterDim;
		update_param _weightUpdateParam;
		update_param _biasUpdateParam;
		param_filler<Dtype> _weightFiller;
		param_filler<Dtype> _biasFiller;
		//typename Activation<Dtype>::Type _activationType;

		Builder() {
			this->type = Layer<Dtype>::Conv;
			_filterDim.rows = 0;
			_filterDim.cols = 0;
			_filterDim.channels = 0;
			_filterDim.filters = 0;
			_filterDim.pad = 0;
			_filterDim.stride = 0;
			_weightUpdateParam.lr_mult = 1.0;
			_weightUpdateParam.decay_mult = 0.0;
			_biasUpdateParam.lr_mult = 1.0;
			_biasUpdateParam.decay_mult = 0.0;
			_weightFiller.type = ParamFillerType::Constant;
			_weightFiller.value = 0.0f;
			_biasFiller.type = ParamFillerType::Constant;
			_biasFiller.value = 0.0f;
			//_activationType = Activation<Dtype>::NoActivation;
		}
		Builder* filterDim(uint32_t rows, uint32_t cols, uint32_t channels, uint32_t filters,
				uint32_t pad, uint32_t stride) {
			_filterDim.rows = rows;
			_filterDim.cols = cols;
			_filterDim.channels = channels;
			_filterDim.filters = filters;
			_filterDim.pad = pad;
			_filterDim.stride = stride;
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
		/*
		Builder* activationType(typename Activation<Dtype>::Type activationType) {
			this->_activationType = activationType;
			return this;
		}
		*/
		virtual Builder* name(const std::string name) {
			HiddenLayer<Dtype>::Builder::name(name);
			return this;
		}
		virtual Builder* id(uint32_t id) {
			HiddenLayer<Dtype>::Builder::id(id);
			return this;
		}
		virtual Builder* inputs(const std::vector<std::string>& inputs) {
			HiddenLayer<Dtype>::Builder::inputs(inputs);
			return this;
		}
		virtual Builder* outputs(const std::vector<std::string>& outputs) {
			HiddenLayer<Dtype>::Builder::outputs(outputs);
			return this;
		}
		virtual Builder* propDown(const std::vector<bool>& propDown) {
			HiddenLayer<Dtype>::Builder::propDown(propDown);
			return this;
		}

		Layer<Dtype>* build() {
			return new ConvLayer(this);
		}

	};
	/**
	 * @details ConvLayer 기본 생성자
	 */
	ConvLayer();
	ConvLayer(Builder* builder);
	/**
	 * @details ConvLayer 생성자
	 * @param filter_d 컨볼루션 연산 관련 파라미터 구조체
	 * @param weight_update_param weight 갱신 관련 파라미터 구조체
	 * @param bias_update_param bias 갱신 관련 파라미터 구조체
	 * @param weight_filler filter 초기화 관련 파라미터 구조체
	 * @param bias_filler bias 초기화 관련 파라미터 구조체
	 * @param activationType 컨볼루션 결과에 적용할 활성화 타입
	 */
	ConvLayer(const std::string name, filter_dim filter_d, update_param weight_update_param, 
              update_param bias_update_param, param_filler<Dtype> weight_filler, 
              param_filler<Dtype> bias_filler);
	/**
	 * @details ConvLayer 소멸자
	 */
	virtual ~ConvLayer();

	/**
	 * @details 컨볼루션 연산 관련 파라미터 구조체를 조회한다.
	 * @return 컨볼루션 연산 관련 파라미터
	 */
	filter_dim &get_filter_dim() { return this->filter_d; }



	//////////////////////////////////////////
	// Learnable Layer Method
	//////////////////////////////////////////
	using HiddenLayer<Dtype>::getName;
	virtual const std::string getName() { return this->name; }
	virtual void update();
	virtual double sumSquareParamsData();
	virtual double sumSquareParamsGrad();
	virtual void scaleParamsGrad(float scale);
	//virtual double testParamAbnormality();
	virtual uint32_t boundParams();
	virtual uint32_t numParams();
	virtual void saveParams(std::ofstream& ofs);
	virtual void loadParams(std::ifstream& ifs);
	virtual void loadParams(std::map<std::string, Data<Dtype>*>& dataMap);
	//////////////////////////////////////////

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();


	
    //
    void syncMutableMem();
    void applyChanges(LearnableLayer<Dtype> *targetLayer);
    void syncParams(LearnableLayer<Dtype> *targetLayer);







protected:
	void initialize(filter_dim filter_d, update_param weight_update_param,
        update_param bias_update_param, param_filler<Dtype> weight_filler,
        param_filler<Dtype> bias_filler);



	void _computeFiltersConvolutionData();
	//void _computeActivationData();

	//void _computePreActivationGrad();
	void _computeFiltersGrad();
	void _computeBiasesGrad();
	void _computeInputGrad();

	void _updateParam(const uint32_t paramSize, const Dtype regScale, const Dtype learnScale,
        Data<Dtype>* dataHistory, Data<Dtype>* data);

	enum ParamType {
		Filter = 0,
		Bias = 1
	};

protected:
	filter_dim filter_d;							///< 컨볼루션 연산 관련 파라미터 구조체
	//Activation<Dtype> *activation_fn;				///< 활성화 객체

	update_param weight_update_param;				///< weight 갱신 관련 파라미터 구조체
	update_param bias_update_param;					///< bias 갱신 관련 파라미터 구조체
	param_filler<Dtype> weight_filler;				///< weight 초기화 관련 파라미터 구조체
	param_filler<Dtype> bias_filler;				///< bias 초기화 관련 파라미터 구조체



#ifndef GPU_MODE
#else
	cudnnTensorDescriptor_t inputTensorDesc;	///< cudnn 입력 데이터(n-D 데이터셋) 구조 정보
	cudnnTensorDescriptor_t outputTensorDesc;	///< cudnn 출력 데이터(n-D 데이터셋) 구조 정보
	cudnnTensorDescriptor_t biasTensorDesc;		///< cudnn bias 구조 정보 구조체
	cudnnFilterDescriptor_t filterDesc;			///< cudnn filter 구조 정보 구조체
	cudnnConvolutionDescriptor_t convDesc;		///< cudnn 컨볼루션 연산 정보 구조체
	cudnnConvolutionFwdAlgo_t convFwdAlgo;		///< cudnn 컨볼루션 포워드 알고리즘 열거형 
	cudnnConvolutionBwdFilterAlgo_t convBwdFilterAlgo;	///< cudnn filter 백워드 열거형
	cudnnConvolutionBwdDataAlgo_t convBwdDataAlgo;	///< cudnn data 백워드 알고리즘 열거형

	size_t workspaceSize;	///< cudnn forward, backward에 필요한 작업공간 GPU 메모리 사이즈
	void *d_workspace;		///< cudnn forward, backward에 필요한 작업공간 장치 메모리 포인터
#endif

public:
	//Data<Dtype>* _preActivation;		    	///< 컨볼루션 결과에 대한 데이터
    std::vector<Data<Dtype>*> _params;			///< 파리미터 데이터 (Filter, Bias 포함)
    std::vector<Data<Dtype>*> _paramsHistory;	///< 이전 update의 파라미터 그레디언트 데이터

    //std::vector<bool> _paramsInitialized;
};

#endif /* LAYER_CONVLAYER_H_ */
