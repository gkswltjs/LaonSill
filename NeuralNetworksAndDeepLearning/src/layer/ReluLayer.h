/*
 * ReluLayer.h
 *
 *  Created on: Jan 25, 2017
 *      Author: jkim
 */

#ifndef RELULAYER_H_
#define RELULAYER_H_

#include "common.h"
#include "BaseLayer.h"

template <typename Dtype>
class ReluLayer : public Layer<Dtype> {
public:
    ReluLayer();
	virtual ~ReluLayer();

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();

protected:
    void applyLeakyForward();
    void applyLeakyBackward();

protected:
	// input, output tensor의 desc가 동일하므로 하나만 사용
	cudnnTensorDescriptor_t tensorDesc;	///< cudnn 입력 데이터(n-D 데이터셋) 구조 정보
	cudnnActivationDescriptor_t activationDesc;	///< cudnn 활성화 관련 자료구조에 대한 포인터

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

#endif /* RELULAYER_H_ */
