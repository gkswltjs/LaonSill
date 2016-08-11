/**
 * @file	ConvLayer.h
 * @date	2016/5/23
 * @author	jhkim
 * @brief
 * @details
 */

#ifndef LAYER_CONVLAYER_H_
#define LAYER_CONVLAYER_H_

#include <string>

#include "../activation/ActivationFactory.h"
#include "../Util.h"
#include "HiddenLayer.h"
#include "LayerConfig.h"





/**
 * @brief 컨볼루션 레이어
 * @details
 */
class ConvLayer : public HiddenLayer {
public:
	ConvLayer() { this->type = LayerType::Conv; }
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
			param_filler weight_filler, param_filler bias_filler, ActivationType activationType);
	virtual ~ConvLayer();

	/**
	 * @details 컨볼루션 연산 관련 파라미터 구조체를 조회한다.
	 * @return 컨볼루션 연산 관련 파라미터
	 */
	filter_dim &get_filter_dim() { return this->filter_d; }
	void backpropagation(UINT idx, DATATYPE *next_delta_input);

	void update(UINT idx, UINT n, UINT miniBatchSize);
	virtual void load(ifstream &ifs, map<Layer *, Layer *> &layerMap);
	virtual bool isLearnable() { return true; }

#ifndef GPU_MODE
public:

	rcube *getWeight() { return this->filters; }
	rcube &getDeltaInput() { return this->delta_input; }


	/**
	 * 주어진 입력 input에 대해 출력 activation을 계산
	 * @param input: 레이어 입력 데이터 (이전 레이어의 activation)
	 */
	void feedforward(UINT idx, const rcube &input, const char *end=0);

#else
	//static void init();
	//static void destroy();

	DATATYPE *getWeight() { return this->filters; }
	DATATYPE *getDeltaInput() { return this->d_delta_input; }
	void feedforward(UINT idx, const DATATYPE *input, const char *end=0);
#endif


protected:
	void initialize(filter_dim filter_d, update_param weight_update_param, update_param bias_update_param,
			param_filler weight_filler, param_filler bias_filler, ActivationType activationType);

	virtual void _save(ofstream &ofs);
	virtual void _shape(bool recursive=true);
	virtual void _clearShape();
	virtual DATATYPE _sumSquareParam();
	virtual DATATYPE _sumSquareParam2();
	virtual void _scaleParam(DATATYPE scale_factor);

	filter_dim filter_d;							///< 컨볼루션 연산 관련 파라미터 구조체
	Activation *activation_fn;						///< 활성화 객체

	update_param weight_update_param;				///< weight 갱신 관련 파라미터 구조체
	update_param bias_update_param;					///< bias 갱신 관련 파라미터 구조체
	param_filler weight_filler;						///< weight 초기화 관련 파라미터 구조체
	param_filler bias_filler;						///< bias 초기화 관련 파라미터 구조체

#ifndef GPU_MODE
protected:
	void convolution(const rmat &x, const rmat &w, rmat &result, int stride);
	void dw_convolution(const rmat &d, const rmat &x, rmat &result);
	void dx_convolution(const rmat &d, const rmat &w, rmat &result);

	rcube *filters;		// weights
	rvec biases;

	rcube *nabla_w;
	rvec nabla_b;

	rcube z;
	rcube delta;
	rcube delta_input;
#else
protected:
	DATATYPE *filters;								///< filter 호스트 메모리 포인터
	DATATYPE *biases;								///< bias 호스트 메모리 포인터

	DATATYPE *d_filters;							///< filter 장치 메모리 포인터
	DATATYPE *d_biases;								///< bias 장치 메모리 포인터

	DATATYPE *d_z;									///< filter map 장치 메모리 포인터
	DATATYPE *d_delta;								///< 네트워크 cost의 z(filter map)에 관한 gradient 장치 메모리 포인터
	DATATYPE *d_delta_weight;						///< 네트워크 cost의 weight(filter)에 관한 gradient 장치 메모리 포인터
	DATATYPE *d_delta_weight_prev;					///< 이전 업데이트의 네트워크 cost의 weight에 관한 graident 장치 메모리 포인터
	DATATYPE *d_delta_bias;							///< 네트워크 cost의 bias에 관한 gradient 장치 메모리 포인터
	DATATYPE *d_delta_bias_prev;					///< 이전 업데이트의 네트워크 cost의 bias에 관한 gradient 장치 메모리 포인터

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
