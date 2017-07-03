/*
 * RoIInputLayer.h
 *
 *  Created on: Nov 11, 2016
 *      Author: jkim
 */

#ifndef ROIINPUTLAYER_H_
#define ROIINPUTLAYER_H_

#include <opencv2/highgui/highgui.hpp>

#include "frcnn_common.h"
#include "DataSet.h"
#include "InputLayer.h"
#include "IMDB.h"
#include "SysLog.h"


template <typename Dtype>
class RoIInputLayer : public InputLayer<Dtype> {
public:
	RoIInputLayer();
	virtual ~RoIInputLayer();

	virtual void feedforward();
	using InputLayer<Dtype>::feedforward;
	virtual void feedforward(const uint32_t baseIndex, const char* end=0);

    int getNumTrainData();
    int getNumTestData();
    void shuffleTrainDataSet();

	void reshape();

private:
	IMDB* getImdb();
	void getTrainingRoidb(IMDB* imdb);
	IMDB* getRoidb();
	IMDB* combinedRoidb();
	bool isValidRoidb(RoIDB& roidb);
	void filterRoidb(std::vector<RoIDB>& roidb);

	void shuffleRoidbInds();
	void getNextMiniBatch();
	void getNextMiniBatchInds(std::vector<uint32_t>& inds);
	void getMiniBatch(const std::vector<RoIDB>& roidb, const std::vector<uint32_t>& inds);
	std::vector<cv::Mat> getImageBlob(const std::vector<RoIDB>& roidb,
			const std::vector<uint32_t>& scaleInds, std::vector<float>& imScales);
	float prepImForBlob(cv::Mat& im, cv::Mat& imResized, const std::vector<float>& pixelMeans,
			const uint32_t targetSize, const uint32_t maxSize);
	void imListToBlob(std::vector<cv::Mat>& ims);


public:
	std::vector<std::vector<float>> bboxMeans;
	std::vector<std::vector<float>> bboxStds;
	IMDB* imdb;

	std::vector<uint32_t> perm;
	uint32_t cur;

	std::vector<std::vector<Data<Dtype>*>> proposalTargetDataList;

	std::vector<cv::Scalar> boxColors;


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

#endif /* ROIINPUTLAYER_H_ */
