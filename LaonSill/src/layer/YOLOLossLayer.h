/**
 * @file YOLOLossLayer.h
 * @date 2017-04-21
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef YOLOLOSSLAYER_H
#define YOLOLOSSLAYER_H 

#include "common.h"
#include "LossLayer.h"
#include "LayerConfig.h"

#define YOLO_GRID_COUNT                 169
#define YOLO_GRID_ONE_AXIS_COUNT        13
#define YOLO_ANCHOR_BOX_COUNT           5
#define YOLO_CLASS_COUNT                20
#define YOLO_ELEM_COUNT_PER_ANCHORBOX   (YOLO_CLASS_COUNT + 5)
#define YOLO_GRID_ELEM_COUNT        (YOLO_ANCHOR_BOX_COUNT * YOLO_ELEM_COUNT_PER_ANCHORBOX)

#define YOLO_GROUND_TRUTH_ELEM_COUNT    (YOLO_CLASS_COUNT + 6)

#define YOLOINPUT_ELEMCOUNT_PER_GRID            7

template<typename Dtype>
class YOLOLossLayer : public LossLayer<Dtype> {
public: 
    YOLOLossLayer();
    virtual ~YOLOLossLayer();

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();
	virtual Dtype cost();

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

#endif /* YOLOLOSSLAYER_H */
