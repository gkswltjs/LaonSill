/**
 * @file AAALayer.h
 * @date 2017-01-25
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef BATCHNORMLAYER_H
#define BATCHNORMLAYER_H 

#include "common.h"
#include "BaseLayer.h"
#include "LearnableLayer.h"
#include "LayerConfig.h"

template <typename Dtype>
class AAALayer : public Layer<Dtype> {
public: 
    AAALayer();
    virtual ~AAALayer();

	virtual void backpropagation();
	virtual void reshape();
	virtual void feedforward();

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
#endif /* BATCHNORMLAYER_H */
