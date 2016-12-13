/**
 * @file ALEInputLayer.cpp
 * @date 2016-12-09
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <stdlib.h>
#include <time.h>

#include "common.h"
#include "ALEInputLayer.h"
#include "NetworkConfig.h"

using namespace std;

template<typename Dtype>
ALEInputLayer<Dtype>::ALEInputLayer() {
    initialize();
}

template<typename Dtype>
ALEInputLayer<Dtype>::ALEInputLayer(const string name) : Layer<Dtype>(name) {
    initialize();
}

template<typename Dtype>
ALEInputLayer<Dtype>::ALEInputLayer(Builder* builder) : Layer<Dtype>(builder) {
	initialize();
}

template<typename Dtype>
ALEInputLayer<Dtype>::~ALEInputLayer() {
    SASSERT0(this->rmSlots != NULL);
    for (int i = 0; i < this->rmSlotCnt; i++) {
        SASSERT0(this->rmSlots[i] != NULL);
        delete this->rmSlots[i];
    }
    free(this->rmSlots);

    SASSERT0(this->stateSlots != NULL);
    for (int i = 0; i < this->stateSlotCnt; i++) {
        SASSERT0(this->stateSlots[i] != NULL);
        delete this->stateSlots[i];
    }
    free(this->stateSlots);

    if (this->preparedData)
        free(this->preparedData);

    if (this->preparedLabel)
        free(this->preparedLabel);
}

template<typename Dtype>
int ALEInputLayer<Dtype>::getInputSize() const {
    return this->in_dim.rows * this->in_dim.cols * this->in_dim.channels;
}

template<typename Dtype>
void ALEInputLayer<Dtype>::shape() {

    SASSERT(this->networkConfig->_batchSize <= this->rmSlotCnt,
        "batch size should be less than replay memory slot count."
        " batch size=%d, replay memory slot count=%d",
        this->networkConfig->_batchSize, this->rmSlotCnt);

    this->in_dim.batches = this->networkConfig->_batchSize;
    this->in_dim.channels = this->chCnt;
    this->in_dim.rows = this->rowCnt;
    this->in_dim.cols = this->colCnt;

    _shape();
}

template<typename Dtype>
void ALEInputLayer<Dtype>::insertFrameInfo(Dtype* img, int action, Dtype reward, bool term) {
    int copySize = this->stateSlots[this->stateSlotHead]->getDataSize();
    memcpy((void*)this->stateSlots[this->stateSlotHead]->data, (void*)img, copySize);

    this->rmSlots[this->rmSlotHead]->fill(
        this->stateSlots[this->rmSlotHead], action, reward, this->lastState, term);

    this->rmSlotHead = (this->rmSlotHead + 1) % this->rmSlotCnt;
    this->stateSlotHead = (this->stateSlotHead + 1) % this->stateSlotCnt;
}

template<typename Dtype>
void ALEInputLayer<Dtype>::prepareInputData() {
    int inputDataCount = this->in_dim.batches;
	const uint32_t unitSize = this->in_dim.unitsize();

    if (this->preparedData == NULL) {
        int allocSize = unitSize * inputDataCount * sizeof(Dtype);
        this->preparedData = (Dtype*)malloc(allocSize);

        SASSERT0(this->preparedData != NULL);
        SASSERT0(this->preparedLabel == NULL);

        allocSize = inputDataCount * sizeof(Dtype);
        this->preparedLabel = (Dtype*)malloc(allocSize);
        SASSERT0(this->preparedLabel != NULL);
    }

    srand(time(NULL));

    for (int i = 0; i < inputDataCount; i++) {
        int index = rand() % this->rmSlotCnt;
        int action = this->rmSlots[index]->action1;

        Dtype reward1 = this->rmSlots[index]->reward1;
        DQNState<Dtype>* state1 = this->rmSlots[index]->state1;
        Dtype maxQ2 = this->rmSlots[index]->maxQ2;

        int preparedDataOffset = i * state1->getDataCount();
        memcpy((void*)&this->preparedData[preparedDataOffset],
            (void*)state1->data, state1->getDataSize());

        int preparedLabelOffset = i; // 1 label
        memcpy((void*)&this->preparedLabel[preparedLabelOffset],
            (void*)&reward1, sizeof(Dtype));
    }
}

template<typename Dtype>
void ALEInputLayer<Dtype>::feedforward() {
    Layer<Dtype>::feedforward();
}

template<typename Dtype>
void ALEInputLayer<Dtype>::feedforward(const uint32_t baseIndex, const char* end) {
	const uint32_t unitSize = this->in_dim.unitsize();
    int batchSize = this->in_dim.batches;

    this->_inputData[0]->set_device_with_host_data(this->preparedData, 0, unitSize * batchSize);
    this->_inputData[1]->set_device_with_host_data(this->preparedLabel, 0, batchSize);

	Layer<Dtype>::feedforward();
}

template<typename Dtype>
void ALEInputLayer<Dtype>::initialize() {
    this->type = Layer<Dtype>::ALEInput;

    this->rowCnt    = SPARAM(ALE_INPUT_ROW_COUNT);
    this->colCnt    = SPARAM(ALE_INPUT_COL_COUNT);
    this->chCnt     = SPARAM(ALE_INPUT_CHANNEL_COUNT);
    this->rmSlotCnt = SPARAM(DQN_REPLAY_MEMORY_ELEM_COUNT);

    int rmSlotAllocSize = sizeof(DQNTransition<Dtype>*) * this->rmSlotCnt;
    this->rmSlots = (DQNTransition<Dtype>**)malloc(rmSlotAllocSize);
    SASSERT0(this->rmSlots != NULL);

    for (int i = 0; i < this->rmSlotCnt; i++) {
        this->rmSlots[i] = new DQNTransition<Dtype>();
        SASSERT0(this->rmSlots[i] != NULL);
    }
    this->rmSlotHead = 0;

    // DQNTransition은 replay memory size만큼 존재 한다.
    // DQNTransition은 연이은 2개의 DQNState로 구성이 된다.
    // 따라서 DQNState는 replay memory size + 1 만큼 존재 한다.
    this->stateSlotCnt = this->rmSlotCnt + 1;

    int stateSlotAllocSize = sizeof(DQNState<Dtype>*) * this->stateSlotCnt;
    this->stateSlots = (DQNState<Dtype>**)malloc(stateSlotAllocSize);

    for (int i = 0; i < this->stateSlotCnt; i++) {
        this->stateSlots[i] = new DQNState<Dtype>(this->rowCnt, this->colCnt, this->chCnt);
        SASSERT0(this->stateSlots[i] != NULL);
    }
    this->stateSlotHead = 0;

    // 아래의 구조체는 초기화(initialize())함수에서 할당하지 않는다.
    // 왜냐하면 networkConfig의 batch size정보가 있어야 할당크기를 정할 수 있는데
    // networkConfig 정보가 채워지는 타이밍이 초기화 이전이기 때문이다.
    // 혹시 바꿀 수 있을지 논의해보자.
    this->preparedData = NULL;
    this->preparedLabel = NULL;

    this->lastState = NULL;
}

template<typename Dtype>
void ALEInputLayer<Dtype>::_shape(bool recursive) {
    this->out_dim = this->in_dim;

    for (uint32_t i = 0; i < this->_outputs.size(); i++) {
		this->_inputs.push_back(this->_outputs[i]);
		this->_inputData.push_back(this->_outputData[i]);
    }

	if (this->_outputs.size() > 1) {
		this->_inputData[1]->shape({this->in_dim.batches, 1, 1, 1});
	}

	if(recursive) {
		Layer<Dtype>::_shape();
	}
}

template<typename Dtype>
void ALEInputLayer<Dtype>::_clearShape() {
    Layer<Dtype>::_clearShape();
}

template class ALEInputLayer<float>;
