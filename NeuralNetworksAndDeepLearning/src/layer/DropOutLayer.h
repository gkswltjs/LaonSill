/**
 * @file DropOutLayer.h
 * @date 2017-04-19
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef DROPOUTLAYER_H
#define DROPOUTLAYER_H 

#include <memory>

#include "common.h"
#include "Layer.h"
#include "LayerConfig.h"
#include "SyncMem.h"

template<typename Dtype>
class DropOutLayer : public Layer<Dtype> {
public: 
	DropOutLayer();
	virtual ~DropOutLayer();

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();

protected:
    void    doDropOutForward();
    void    doDropOutBackward();

    std::shared_ptr<SyncMem<Dtype>>  mask;

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
#endif /* DROPOUTLAYER_H */
