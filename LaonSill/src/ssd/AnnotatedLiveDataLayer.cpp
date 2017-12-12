/*
 * AnnotatedLiveDataLayer.cpp
 *
 *  Created on: Sep 4, 2017
 *      Author: jkim
 */

#include <opencv2/highgui/highgui.hpp>

#include "AnnotatedLiveDataLayer.h"
#include "PropMgmt.h"
#include "SysLog.h"
#include "IO.h"
#include "WorkContext.h"
#include "Param.h"
#include "Perf.h"
#include "EnumDef.h"
#include "MathFunctions.h"
#include "Sampler.h"
#include "ImageUtil.h"
#include "MemoryMgmt.h"

using namespace std;



template <typename Dtype>
AnnotatedLiveDataLayer<Dtype>::AnnotatedLiveDataLayer()
: InputLayer<Dtype>(),
  dataTransformer(&SLPROP(AnnotatedLiveData, dataTransformParam)),
  videoCapture(SLPROP(AnnotatedLiveData, camIndex)) {
	this->type = Layer<Dtype>::AnnotatedLiveData;

	DataTransformParam& dataTransformParam = this->dataTransformer.param;
	dataTransformParam.resizeParam = SLPROP(AnnotatedLiveData, resizeParam);
	dataTransformParam.resizeParam.updateInterpMode();

	// Make sure dimension is consistent within batch.
	if (this->dataTransformer.param.resizeParam.prob >= 0.f) {
		if (this->dataTransformer.param.resizeParam.resizeMode == ResizeMode::FIT_SMALL_SIZE) {
			SASSERT(SNPROP(batchSize) == 1, "Only support batch size of 1 for FIT_SMALL_SIZE.");
		}
	}

	SASSERT(this->videoCapture.isOpened(), "video device is not opened ... ");
}

template <typename Dtype>
AnnotatedLiveDataLayer<Dtype>::~AnnotatedLiveDataLayer() {}

template <typename Dtype>
void AnnotatedLiveDataLayer<Dtype>::reshape() {
	if (this->_inputData.size() < 1) {
		for (uint32_t i = 0; i < SLPROP(AnnotatedLiveData, output).size(); i++) {
			SLPROP(AnnotatedLiveData, input).push_back(SLPROP(AnnotatedLiveData, output)[i]);
			this->_inputData.push_back(this->_outputData[i]);
		}
	}
	Layer<Dtype>::_adjustInputShape();

    const int batchSize = SNPROP(batchSize);



    /*
    // XXX: trace하면서 확인.
    // 여기는 mock data로 임시로 넘어가도록 수정
    AnnotatedDatum* annoDatum = new AnnotatedDatum();
    annoDatum->encoded;
    annoDatum->channels = 3;

    // Use data transformer to infer the expected data shape from annoDatum.
    vector<uint32_t> outputShape = this->dataTransformer.inferDataShape(annoDatum);
    outputShape[0] = batchSize;
    this->_outputData[0]->reshape(outputShape);
    */

}

template <typename Dtype>
void AnnotatedLiveDataLayer<Dtype>::feedforward() {
	reshape();
	load_batch();
}

template <typename Dtype>
void AnnotatedLiveDataLayer<Dtype>::feedforward(unsigned int baseIndex, const char* end) {
	reshape();
	load_batch();
}



template <typename Dtype>
void AnnotatedLiveDataLayer<Dtype>::load_batch() {
	cv::Mat frame;
	bool frameValid = false;

	while (!frameValid) {
		try {
			this->videoCapture >> frame;
			frameValid = true;
		} catch (cv::Exception& e) {
			cout << "frame invalid ...  skip current frame ... " << endl;
		}
	}


	const ResizeParam& resizeParam = this->dataTransformer.param.resizeParam;
	cv::resize(frame, frame, cv::Size(resizeParam.width, resizeParam.height),
			0, 0, cv::INTER_LINEAR);

	// DataTransformer::transform에서 이미지 정보를 CHW 기준으로 조회해서
	// 아래와 같이 shape 생성
	vector<uint32_t> dataShape = {1, frame.channels(), frame.rows, frame.cols};
	//vector<uint32_t> dataShape = {1, frame.rows, frame.cols, frame.channels()};
	this->_outputData[0]->reshape(dataShape);

	Datum datum;
	// channel_separated false -> opencv의 (b, g, r), (b, g, r) ... 을 그대로 저장
	// channel_separated true  -> 실상황 데이터를 전송하기 위해
	CVMatToDatum(frame, true, &datum);
	this->dataTransformer.transform(&datum, this->_outputData[0], 0);

	//this->_printOn();
	//this->_outputData[0]->print_data({}, false, -1);
	//this->_printOff();

	/*
	Dtype* outputData = this->_outputData[0]->mutable_host_data();

	const int batchSize = SNPROP(batchSize);
	SASSERT0(batchSize == 1);

	for (int itemId = 0; itemId < batchSize; itemId++) {
		// XXX: 카메라 프레임인 cv::Mat으로부터 AnnoatedDatum으로 변환
		// 또는 AnnotatedDatum에 대해 cv::Mat에 대응하는 함수 코드 필요
		AnnotatedDatum* annoDatum;
		this->dataTransformer.transform(annoDatum, this->_outputData[0], itemId);
	}
	*/
}














template <typename Dtype>
int AnnotatedLiveDataLayer<Dtype>::getNumTrainData() {
	// XXX: int max로 설정해 두면 될 듯
	return INT_MAX;
}

template <typename Dtype>
int AnnotatedLiveDataLayer<Dtype>::getNumTestData() {
	return 0;
}

template <typename Dtype>
void AnnotatedLiveDataLayer<Dtype>::shuffleTrainDataSet() {

}








/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* AnnotatedLiveDataLayer<Dtype>::initLayer() {
	AnnotatedLiveDataLayer* layer = NULL;
	SNEW(layer, AnnotatedLiveDataLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void AnnotatedLiveDataLayer<Dtype>::destroyLayer(void* instancePtr) {
    AnnotatedLiveDataLayer<Dtype>* layer = (AnnotatedLiveDataLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void AnnotatedLiveDataLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
	// XXX
	SASSERT0(!isInput);
	SASSERT0(index < 1);

    AnnotatedLiveDataLayer<Dtype>* layer = (AnnotatedLiveDataLayer<Dtype>*)instancePtr;
	SASSERT0(layer->_outputData.size() == index);
	layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
}

template<typename Dtype>
bool AnnotatedLiveDataLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    AnnotatedLiveDataLayer<Dtype>* layer = (AnnotatedLiveDataLayer<Dtype>*)instancePtr;
    layer->reshape();

    if (SNPROP(miniBatch) == 0) {
		int trainDataNum = layer->getNumTrainData();
		if (trainDataNum % SNPROP(batchSize) == 0) {
			SNPROP(miniBatch) = trainDataNum / SNPROP(batchSize);
		} else {
			SNPROP(miniBatch) = trainDataNum / SNPROP(batchSize) + 1;
		}
		WorkContext::curPlanInfo->miniBatchCount = SNPROP(miniBatch);
	}

    return true;
}

template<typename Dtype>
void AnnotatedLiveDataLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
	AnnotatedLiveDataLayer<Dtype>* layer = (AnnotatedLiveDataLayer<Dtype>*)instancePtr;
	layer->feedforward();

}

template<typename Dtype>
void AnnotatedLiveDataLayer<Dtype>::backwardTensor(void* instancePtr) {
    // do nothing
}

template<typename Dtype>
void AnnotatedLiveDataLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template class AnnotatedLiveDataLayer<float>;
