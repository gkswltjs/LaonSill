/*
 * InputLayer.cpp
 *
 *  Created on: 2016. 9. 12.
 *      Author: jhkim
 */


#include "InputLayer.h"
#include "Network.h"
#include "ImagePackDataSet.h"
#include "MockDataSet.h"
#include "Util.h"
#include "CudaUtils.h"
#include "SysLog.h"
#include "PropMgmt.h"

#define INPUTLAYER_LOG 1

using namespace std;

template <typename Dtype>
InputLayer<Dtype>::InputLayer()
: Layer<Dtype>() {
	this->type = Layer<Dtype>::Input;

    this->_dataSet = NULL;
    this->_dataMean = new Data<Dtype>("dataMean");

    const string& sourceType = SLPROP(Input, sourceType);
	if (sourceType == "ImagePack") {
		const string& source = SLPROP(Input, source);
		const uint32_t numTrainPack = SLPROP(Input, numTrainPack);
		const uint32_t numTestPack = SLPROP(Input, numTestPack);
		this->_dataSet = new ImagePackDataSet<Dtype>(
				source + "/train_data", source + "/train_label", numTrainPack,
				source + "/test_data", source + "/test_label", numTestPack);

		const vector<float>& mean = SLPROP(Input, mean);
		int numChannels = mean.size();
		this->_dataMean->reshape({1, 1, 1, numChannels});
		for (int i = 0; i < numChannels; i++) {
			this->_dataMean->mutable_host_data()[i] = mean[i];
		}
		this->_dataSet->load();
	} else if (sourceType == "Mock") {
		this->_dataSet = new MockDataSet<Dtype>(4, 4, 3, 10, 10, 10);
		this->_dataSet->load();
	}
}


template <typename Dtype>
InputLayer<Dtype>::~InputLayer() {
	if (this->_dataMean) {
		delete this->_dataMean;
	}
}


template <typename Dtype>
void InputLayer<Dtype>::reshape() {
	// 입력 레이어는 출력 데이터를 입력 데이터와 공유
	// xxx: 레이어 명시적으로 초기화할 수 있는 위치를 만들어 옮길 것.
	const vector<string>& outputs = SLPROP(Input, output);
	vector<string>& inputs = SLPROP(Input, input);
	if (inputs.size() < 1) {
		for (uint32_t i = 0; i < outputs.size(); i++) {
			inputs.push_back(outputs[i]);
			this->_inputData.push_back(this->_outputData[i]);
		}
	}
	Layer<Dtype>::_adjustInputShape();

	const uint32_t inputSize = this->_inputData.size();
	for (uint32_t i = 0; i < inputSize; i++) {
		if (!Layer<Dtype>::_isInputShapeChanged(i))
			continue;

		// 데이터
		if (i == 0) {
			uint32_t batches = SNPROP(batchSize);
			uint32_t channels = this->_dataSet->getChannels();
			uint32_t rows = this->_dataSet->getRows();
			uint32_t cols = this->_dataSet->getCols();

			this->_inputShape[0][0] = batches;
			this->_inputShape[0][1] = channels;
			this->_inputShape[0][2] = rows;
			this->_inputShape[0][3] = cols;

			this->_inputData[0]->reshape(this->_inputShape[0]);
		}
		// 레이블
		else if (i == 1) {
			uint32_t batches = SNPROP(batchSize);
			uint32_t channels = 1;
			uint32_t rows = 1;
			uint32_t cols = 1;

			this->_inputShape[1][0] = batches;
			this->_inputShape[1][1] = channels;
			this->_inputShape[1][2] = rows;
			this->_inputShape[1][3] = cols;

			this->_inputData[1]->reshape({batches, channels, rows, cols});
		}
	}
}

template <typename Dtype>
void InputLayer<Dtype>::feedforward() {
	reshape();
	//cout << "unsupported ... " << endl;
	//exit(1);
	// do nothing
}


template <typename Dtype>
void InputLayer<Dtype>::feedforward(const uint32_t baseIndex, const char* end) {
	reshape();

	const vector<uint32_t>& inputShape = this->_inputShape[0];
	const uint32_t batches = inputShape[0];
	const uint32_t unitSize = Util::vecCountByAxis(inputShape, 1);

	NetworkStatus status = SNPROP(status);
	if (status == NetworkStatus::Train) {
		// data
		for (uint32_t i = 0; i < batches; i++) {
			const Dtype* ptr = this->_dataSet->getTrainDataAt(baseIndex+i);
			this->_inputData[0]->set_device_with_host_data(ptr, i*unitSize, unitSize);
		}
		this->_inputData[0]->scale_device_data(SLPROP(Input, scale));

		const uint32_t singleChannelSize = this->_inputData[0]->getCountByAxis(2);
		const Dtype* mean = this->_dataMean->device_data();

		for (uint32_t i = 0; i < batches; i++) {
			Dtype* data = this->_inputData[0]->mutable_device_data() + i*unitSize;
			soooa_sub_channel_mean(unitSize, singleChannelSize, mean, data);
		}

		// label
		if (this->_inputData.size() > 1) {
			for (uint32_t i = 0; i < batches; i++) {
				const Dtype* ptr = this->_dataSet->getTrainLabelAt(baseIndex+i);
				this->_inputData[1]->set_device_with_host_data(ptr, i, 1);
			}
		}
	} else if (status == NetworkStatus::Test) {
		for(uint32_t i = 0; i < batches; i++) {
			const Dtype* ptr = this->_dataSet->getTestDataAt(baseIndex+i);
			this->_inputData[0]->set_device_with_host_data(ptr, i*unitSize, unitSize);
		}

		if (this->_inputData.size() > 1) {
			for (uint32_t i = 0; i < batches; i++) {
				const Dtype* ptr = this->_dataSet->getTestLabelAt(baseIndex+i);
				this->_inputData[1]->set_device_with_host_data(ptr, i, 1);
			}
		}
	} else {
        SASSERT(false, "Invalid network status =%d", status);
    }
}

template<typename Dtype>
int InputLayer<Dtype>::getNumTrainData() {
    return this->_dataSet->getNumTrainData();
}

template<typename Dtype>
int InputLayer<Dtype>::getNumTestData() {
    return this->_dataSet->getNumTestData();
}

template<typename Dtype>
void InputLayer<Dtype>::shuffleTrainDataSet() {
    return this->_dataSet->shuffleTrainDataSet();
}

/****************************************************************************
 * layer callback functions 
 ****************************************************************************/
template<typename Dtype>
void* InputLayer<Dtype>::initLayer() {
    InputLayer* layer = new InputLayer<Dtype>();
    return (void*)layer;
}

template<typename Dtype>
void InputLayer<Dtype>::destroyLayer(void* instancePtr) {
    InputLayer<Dtype>* layer = (InputLayer<Dtype>*)instancePtr;
    delete layer;
}

template<typename Dtype>
void InputLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {

    InputLayer<Dtype>* layer = (InputLayer<Dtype>*)instancePtr;

    SASSERT0(!isInput);

    SASSERT0(layer->_outputData.size() == index);
    layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
}

template<typename Dtype>
bool InputLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    InputLayer<Dtype>* layer = (InputLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void InputLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
	InputLayer<Dtype>* layer = (InputLayer<Dtype>*)instancePtr;
	layer->feedforward();
}

template<typename Dtype>
void InputLayer<Dtype>::backwardTensor(void* instancePtr) {
	SASSERT0(false);
}

template<typename Dtype>
void InputLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template class InputLayer<float>;
