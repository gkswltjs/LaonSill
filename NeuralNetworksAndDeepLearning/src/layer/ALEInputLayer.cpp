/**
 * @file ALEInputLayer.cpp
 * @date 2016-12-09
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "common.h"
#include "ALEInputLayer.h"
#include "InputLayer.h"
#include "NetworkConfig.h"
#include "PropMgmt.h"
#include "SysLog.h"

using namespace std;

template<typename Dtype>
ALEInputLayer<Dtype>::ALEInputLayer(Builder* builder) : InputLayer<Dtype>(builder) {
	initialize();
}

template<typename Dtype>
ALEInputLayer<Dtype>::ALEInputLayer(const string& name) : InputLayer<Dtype>(name) {
	initialize();
}

template<typename Dtype>
ALEInputLayer<Dtype>::~ALEInputLayer() {
    if (this->preparedData)
        free(this->preparedData);

    if (this->preparedLabel)
        free(this->preparedLabel);
}

template<typename Dtype>
void ALEInputLayer<Dtype>::setInputCount(int rows, int cols, int channels, int actions) {
    this->rowCnt = rows;
    this->colCnt = cols;
    this->chCnt = channels;
    this->actionCnt = actions;
}

template <typename Dtype>
void ALEInputLayer<Dtype>::reshape() {
	// 입력 레이어는 출력 데이터를 입력 데이터와 공유
	// xxx: 레이어 명시적으로 초기화할 수 있는 위치를 만들어 옮길 것.
	if (this->_inputData.size() < 1) {
		for (uint32_t i = 0; i < SLPROP_BASE(output).size(); i++) {
			SLPROP_BASE(input).push_back(SLPROP_BASE(output)[i]);
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
			uint32_t channels = this->chCnt;
			uint32_t rows = this->rowCnt;
			uint32_t cols = this->colCnt;

			this->_inputShape[0][0] = batches;
			this->_inputShape[0][1] = channels;
			this->_inputShape[0][2] = rows;
			this->_inputShape[0][3] = cols;

			this->_inputData[0]->reshape(this->_inputShape[0]);

#if INPUTLAYER_LOG
			printf("<%s> layer' output-0 has reshaped as: %dx%dx%dx%d\n",
					this->name.c_str(), batches, channels, rows, cols);
#endif
		}
		// 레이블
		else if (i == 1) {
			uint32_t batches = SNPROP(batchSize);
			uint32_t channels = this->chCnt;
			uint32_t rows = 1;
			uint32_t cols = 1;

			this->_inputShape[1][0] = batches;
			this->_inputShape[1][1] = channels;
			this->_inputShape[1][2] = rows;
			this->_inputShape[1][3] = cols;

			this->_inputData[1]->reshape({batches, channels, rows, cols});

#if INPUTLAYER_LOG
			printf("<%s> layer' output-1 has reshaped as: %dx%dx%dx%d\n",
					this->name.c_str(), batches, channels, rows, cols);
#endif
		}
	}
}

template<typename Dtype>
void ALEInputLayer<Dtype>::allocInputData() {
	const vector<uint32_t>& inputShape = this->_inputShape[0];
	const uint32_t batchSize = inputShape[0];
	const uint32_t unitSize = Util::vecCountByAxis(inputShape, 1);

    if (batchSize > this->allocBatchSize) {
        if (this->preparedData != NULL) {
            free(this->preparedData);
            free(this->preparedLabel);
        }

        this->allocBatchSize = batchSize;

        int allocSize = unitSize * this->allocBatchSize * sizeof(Dtype);
        this->preparedData = (Dtype*)malloc(allocSize);
        SASSERT0(this->preparedData != NULL);

        allocSize = this->allocBatchSize * this->actionCnt * sizeof(Dtype);
        this->preparedLabel = (Dtype*)malloc(allocSize);
        SASSERT0(this->preparedLabel != NULL);
    }
}

template<typename Dtype>
void ALEInputLayer<Dtype>::feedforward() {
	//Layer<Dtype>::feedforward();
	cout << "unsupported ... " << endl;
	exit(1);
}

template<typename Dtype>
void ALEInputLayer<Dtype>::feedforward(const uint32_t baseIndex, const char* end) {
    reshape();

	const vector<uint32_t>& inputShape = this->_inputShape[0];
	const uint32_t batchSize = inputShape[0];
	const uint32_t unitSize = Util::vecCountByAxis(inputShape, 1);

    this->_inputData[0]->set_device_with_host_data(this->preparedData, 0,
        unitSize * batchSize);
    this->_inputData[1]->set_device_with_host_data(this->preparedLabel, 0,
        this->chCnt * batchSize);
}

template<typename Dtype>
void ALEInputLayer<Dtype>::initialize() {
    this->type = Layer<Dtype>::ALEInput;

    this->preparedData = NULL;
    this->preparedLabel = NULL;
    this->allocBatchSize = 0;
}

template<typename Dtype>
void ALEInputLayer<Dtype>::fillData(DQNImageLearner<Dtype> *learner, bool useState1) {
	const vector<uint32_t>& inputShape = this->_inputShape[0];
	const uint32_t batchSize = inputShape[0];
    int inputDataCount = batchSize;

    allocInputData();
    DQNTransition<Dtype> **rmSlots = learner->getActiveRMSlots();
    for (int i = 0; i < inputDataCount; i++) {
        DQNTransition<Dtype> *rmSlot = rmSlots[i];

        int action = rmSlot->action1;
        Dtype reward1 = rmSlot->reward1;
        DQNState<Dtype>* state;
        if (useState1)
            state = rmSlot->state1;
        else
            state = rmSlot->state2;

        int dataIndex = i * state->getDataCount();
        memcpy((void*)&this->preparedData[dataIndex], (void*)state->data,
            state->getDataSize());
    }
}

template<typename Dtype>
void ALEInputLayer<Dtype>::fillLabel(DQNImageLearner<Dtype> *learner) {
	const vector<uint32_t>& inputShape = this->_inputShape[0];
	const uint32_t batchSize = inputShape[0];
    int inputDataCount = batchSize;

    Dtype zero = 0.0;
    Dtype *qLabelValues = learner->getQLabelValues();
    DQNTransition<Dtype> **rmSlots = learner->getActiveRMSlots();

    for (int i = 0; i < inputDataCount; i++) {
        DQNTransition<Dtype> *rmSlot = rmSlots[i];
        Dtype qLabelValue = qLabelValues[i];

        int action = rmSlot->action1;

        // Label데이터를 action으로 부터 생성한다.
        for (int j = 0; j < this->actionCnt; j++) {
            int labelIndex = this->actionCnt * i + j;

            if (j == action) {
                memcpy((void*)&this->preparedLabel[labelIndex], (void*)&qLabelValue,
                    sizeof(Dtype));
            } else {
                memcpy((void*)&this->preparedLabel[labelIndex], (void*)&zero, sizeof(Dtype));
            }
        }
    }
}

template<typename Dtype>
int ALEInputLayer<Dtype>::getNumTrainData() {
    if (this->_dataSet != NULL) {
        return this->_dataSet->getNumTrainData();
    } else {    
        uint32_t batches = SNPROP(batchSize);
        return batches;
    }
}

template<typename Dtype>
int ALEInputLayer<Dtype>::getNumTestData() {
    if (this->_dataSet != NULL) {
        return this->_dataSet->getNumTestData();
    } else {
        return 1;
    }
}

template<typename Dtype>
void ALEInputLayer<Dtype>::shuffleTrainDataSet() {
    if (this->_dataSet != NULL) {
        return this->_dataSet->shuffleTrainDataSet();
    }
}

/****************************************************************************
 * layer callback functions 
 ****************************************************************************/
template<typename Dtype>
void* ALEInputLayer<Dtype>::initLayer() {
    ALEInputLayer* layer = new ALEInputLayer<Dtype>(SLPROP_BASE(name));
    return (void*)layer;
}

template<typename Dtype>
void ALEInputLayer<Dtype>::destroyLayer(void* instancePtr) {
    ALEInputLayer<Dtype>* layer = (ALEInputLayer<Dtype>*)instancePtr;
    delete layer;
}

template<typename Dtype>
void ALEInputLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
    ALEInputLayer<Dtype>* layer = (ALEInputLayer<Dtype>*)instancePtr;

    SASSERT0(!isInput);
    SASSERT0(index < 2);
    SASSERT0(layer->_outputData.size() == index);
    layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
}

template<typename Dtype>
bool ALEInputLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    ALEInputLayer<Dtype>* layer = (ALEInputLayer<Dtype>*)instancePtr;
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
void ALEInputLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
    ALEInputLayer<Dtype>* layer = (ALEInputLayer<Dtype>*)instancePtr;
    layer->feedforward(miniBatchIdx);
}

template<typename Dtype>
void ALEInputLayer<Dtype>::backwardTensor(void* instancePtr) {
    ALEInputLayer<Dtype>* layer = (ALEInputLayer<Dtype>*)instancePtr;
    layer->backpropagation();
}

template<typename Dtype>
void ALEInputLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template class ALEInputLayer<float>;
