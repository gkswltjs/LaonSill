/**
 * @file LiveDataInputLayer.h
 * @date 2017-12-11
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef DATALIVEINPUTLAYER_H
#define DATALIVEINPUTLAYER_H 

#include "InputLayer.h"
#include "Datum.h"

template <typename Dtype>
class LiveDataInputLayer : public InputLayer<Dtype> {
public: 
    LiveDataInputLayer();
    virtual ~LiveDataInputLayer();

	virtual void feedforward();
	using InputLayer<Dtype>::feedforward;
	virtual void feedforward(const uint32_t baseIndex, const char* end=0);

    int getNumTrainData();
    int getNumTestData();
    void shuffleTrainDataSet();

	virtual void reshape();

    void feedImage(const int channels, const int height, const int width, float* image);

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

#endif /* DATALIVEINPUTLAYER_H */
