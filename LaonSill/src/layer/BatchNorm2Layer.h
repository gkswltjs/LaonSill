/*
 * BatchNorm2Layer.h
 *
 *  Created on: Dec 18, 2017
 *      Author: jkim
 */

#ifndef BATCHNORM2LAYER_H_
#define BATCHNORM2LAYER_H_

#include "common.h"
#include "LearnableLayer.h"

template <typename Dtype>
class BatchNorm2Layer : public LearnableLayer<Dtype> {
public:
	BatchNorm2Layer();
	virtual ~BatchNorm2Layer();

	virtual void backpropagation();
	virtual void reshape();
	virtual void feedforward();

	virtual void update();
	void applyChanges(LearnableLayer<Dtype> *targetLayer);
	void syncParams(LearnableLayer<Dtype> *targetLayer);
    bool hasScaleBias() { return this->scaleBias; }

private:
	void multicast_gpu(int N, int C, int S, const Dtype *x, Dtype *y );
	void compute_sum_per_channel_gpu(int N, int C, int S, const Dtype *x, Dtype *y);
	void compute_mean_per_channel_gpu(int N, int C, int S, const Dtype *x, Dtype *y);

private:
	double movingAverageFraction;
	double eps;
	int channels;
	int iter;
	bool useGlobalStats;
	bool clipVariance;
	bool scaleBias;

	std::vector<update_param> updatePolicies;

	Data<Dtype>* mean;
	Data<Dtype>* var;
	Data<Dtype>* invVar;
	Data<Dtype>* xNorm;
	Data<Dtype>* onesN;
	Data<Dtype>* onesHW;
	Data<Dtype>* onesC;
	Data<Dtype>* tempC;
	Data<Dtype>* tempNC;
	Data<Dtype>* tempNCHW;

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


#endif /* BATCHNORM2LAYER_H_ */
