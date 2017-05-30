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
#include "Layer.h"
#include "LearnableLayer.h"
#include "LayerConfig.h"

/**
 * @brief 컨볼루션 레이어
 * @details
 */
template <typename Dtype>
class ConvLayer : public LearnableLayer<Dtype> {
public:
	ConvLayer();
	virtual ~ConvLayer();

	/**
	 * @details 컨볼루션 연산 관련 파라미터 구조체를 조회한다.
	 * @return 컨볼루션 연산 관련 파라미터
	 */
	filter_dim &get_filter_dim() { return this->filter_d; }

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();
	
    //
	virtual void update();
    void applyChanges(LearnableLayer<Dtype> *targetLayer);
    void syncParams(LearnableLayer<Dtype> *targetLayer);

protected:
	void _computeFiltersConvolutionData();
	void _computeFiltersGrad();
	void _computeBiasesGrad();
	void _computeInputGrad();

    // FIXME: 파라미터가 너무 많다. 구조화해서 줄이자.
	void _updateParam(const uint32_t paramSize, const Dtype regScale, const Dtype learnScale,
        const Dtype epsilon, const Dtype decayRate, const Dtype beta1, const Dtype beta2,
        Data<Dtype>* dataHistory, Data<Dtype>* dataHistory2, Data<Dtype>* data);

	enum ParamType {
		Filter = 0,
		Bias = 1
	};

protected:
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


public:
    bool deconv;
    int deconvExtraCell;
    void donateParam(ConvLayer<Dtype>* receiver);

public:
    /****************************************************************************
     * layer callback functions 
     ****************************************************************************/
    static void* initLayer();
    static void destroyLayer(void* instancePtr);
    static void setInOutTensor(void* instancePtr, void* tensorPtr, bool isInput, int index);
    static bool allocLayerTensors(void* instancePtr);
    static void forwardTensor(void* instancePtr, int miniBatchIndex);
    static void backwardTensor(void* instancePtr);
    static void learnTensor(void* instancePtr);
};

#endif /* LAYER_CONVLAYER_H_ */
