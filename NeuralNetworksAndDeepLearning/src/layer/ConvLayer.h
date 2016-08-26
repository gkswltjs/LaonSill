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

#include "../activation/Activation.h"
#include "../Util.h"
#include "HiddenLayer.h"
#include "LearnableLayer.h"
#include "LayerConfig.h"





/**
 * @brief 컨볼루션 레이어
 * @details
 */
class ConvLayer : public HiddenLayer, public LearnableLayer {
public:
	class Builder : public HiddenLayer::Builder {
	public:
		filter_dim _filterDim;
		update_param _weightUpdateParam;
		update_param _biasUpdateParam;
		param_filler _weightFiller;
		param_filler _biasFiller;
		Activation::Type _activationType;

		Builder() {
			_filterDim.rows = 0;
			_filterDim.cols = 0;
			_filterDim.channels = 0;
			_filterDim.filters = 0;
			_filterDim.stride = 0;
			_weightUpdateParam.lr_mult = 1.0;
			_weightUpdateParam.decay_mult = 0.0;
			_biasUpdateParam.lr_mult = 1.0;
			_biasUpdateParam.decay_mult = 0.0;
			_weightFiller.type = ParamFillerType::Constant;
			_biasFiller.type = ParamFillerType::Constant;
			_activationType = Activation::NoActivation;
		}
		Builder* filterDim(uint32_t rows, uint32_t cols, uint32_t channels, uint32_t filters, uint32_t stride) {
			_filterDim.rows = rows;
			_filterDim.cols = cols;
			_filterDim.channels = channels;
			_filterDim.filters = filters;
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
	ConvLayer(const string name, filter_dim filter_d, update_param weight_update_param, update_param bias_update_param,
			param_filler weight_filler, param_filler bias_filler, Activation::Type activationType);
	/**
	 * @details ConvLayer 소멸자
	 */
	virtual ~ConvLayer();



	/**
	 * @details 컨볼루션 연산 관련 파라미터 구조체를 조회한다.
	 * @return 컨볼루션 연산 관련 파라미터
	 */
	filter_dim &get_filter_dim() { return this->filter_d; }







protected:
	void initialize(filter_dim filter_d, update_param weight_update_param, update_param bias_update_param,
			param_filler weight_filler, param_filler bias_filler, Activation::Type activationType);

	virtual void _shape(bool recursive=true);
	virtual void _clearShape();
	//virtual double _sumSquareGrad();
	//virtual double _sumSquareParam();
	virtual double sumSquareParamsData();
	virtual double sumSquareParamsGrad();
	virtual void _save(ofstream &ofs);
	virtual void _load(ifstream &ifs, map<Layer *, Layer *> &layerMap);
	virtual void update();
	virtual void scaleParamsGrad(DATATYPE scale);
	virtual void _feedforward();
	virtual void _backpropagation();

	enum ParamType {
		Filter = 0,
		Bias = 1
	};

protected:
	filter_dim filter_d;							///< 컨볼루션 연산 관련 파라미터 구조체
	Activation *activation_fn;						///< 활성화 객체

	update_param weight_update_param;				///< weight 갱신 관련 파라미터 구조체
	update_param bias_update_param;					///< bias 갱신 관련 파라미터 구조체
	param_filler weight_filler;						///< weight 초기화 관련 파라미터 구조체
	param_filler bias_filler;						///< bias 초기화 관련 파라미터 구조체

#ifndef GPU_MODE
	rcube *filters;		// weights
	rvec biases;

	rcube *nabla_w;
	rvec nabla_b;

	rcube z;
	rcube delta;
	rcube delta_input;
#else
	//DATATYPE *filters;								///< filter 호스트 메모리 포인터
	//DATATYPE *biases;								///< bias 호스트 메모리 포인터

	//DATATYPE *d_filters;							///< filter 장치 메모리 포인터
	//DATATYPE *d_biases;								///< bias 장치 메모리 포인터

	//DATATYPE *d_z;									///< filter map 장치 메모리 포인터
	//DATATYPE *d_delta;								///< 네트워크 cost의 z(filter map)에 관한 gradient 장치 메모리 포인터
	//DATATYPE *d_delta_weight;						///< 네트워크 cost의 weight(filter)에 관한 gradient 장치 메모리 포인터
	//DATATYPE *d_delta_weight_prev;					///< 이전 업데이트의 네트워크 cost의 weight에 관한 graident 장치 메모리 포인터
	//DATATYPE *d_delta_bias;							///< 네트워크 cost의 bias에 관한 gradient 장치 메모리 포인터
	//DATATYPE *d_delta_bias_prev;					///< 이전 업데이트의 네트워크 cost의 bias에 관한 gradient 장치 메모리 포인터

	Data* _preActivation;
	vector<Data*> _params;
	vector<Data*> _paramsHistory;

	cudnnTensorDescriptor_t biasTensorDesc;			///< cudnn bias 구조 정보 구조체
	cudnnFilterDescriptor_t filterDesc;				///< cudnn filter 구조 정보 구조체
	cudnnConvolutionDescriptor_t convDesc;			///< cudnn 컨볼루션 연산 정보 구조체
	cudnnConvolutionFwdAlgo_t convFwdAlgo;			///< cudnn 컨볼루션 포워드 알고리즘 열거형 (입력 데이터에 대해 convolution 수행할 알고리즘)
	cudnnConvolutionBwdFilterAlgo_t convBwdFilterAlgo;	///< cudnn filter 백워드 알고리즘 열거형 (네트워크 cost의 filter에 관한 gradient를 구할 때의 알고리즘)
	cudnnConvolutionBwdDataAlgo_t convBwdDataAlgo;	///< cudnn data 백워드 알고리즘 열거형 (네트워크 cost의 입력 데이터에 관한 gradient를 구할 때의 알고리즘)

	size_t workspaceSize;							///< cudnn forward, backward에 필요한 작업공간 GPU 메모리 사이즈
	void *d_workspace;								///< cudnn forward, backward에 필요한 작업공간 장치 메모리 포인터
#endif






};



#endif /* LAYER_CONVLAYER_H_ */
