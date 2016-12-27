/**
 * @file ALEInputLayer.cpp
 * @date 2016-12-09
 * @author moonhoen lee
 * @brief 
 * @details
 */

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
void ALEInputLayer<Dtype>::setInputCount(int rows, int cols, int channels) {
    this->rowCnt = rows;
    this->colCnt = cols;
    this->chCnt = channels;
}

template<typename Dtype>
void ALEInputLayer<Dtype>::shape() {
	this->in_dim.batches = this->networkConfig->_batchSize;
    this->in_dim.channels = this->chCnt;
    this->in_dim.rows = this->rowCnt;
    this->in_dim.cols = this->colCnt;

    cout << "rows:" << this->in_dim.rows << ",cols:" << this->in_dim.cols <<
        ",channels:" << this->in_dim.channels << ",batches:" << this->in_dim.batches
        << endl;

    _shape();
}

template<typename Dtype>
void ALEInputLayer<Dtype>::allocInputData() {
    int batchSize = this->in_dim.batches;
	const uint32_t unitSize = this->in_dim.unitsize();

    if (batchSize > this->allocBatchSize) {
        if (this->preparedData != NULL) {
            free(this->preparedData);
            free(this->preparedLabel);
        }

        this->allocBatchSize = batchSize;

        int allocSize = unitSize * this->allocBatchSize * sizeof(Dtype);
        cout << "[DEBUG] allocSize = " << allocSize << endl;
        this->preparedData = (Dtype*)malloc(allocSize);
        SASSERT0(this->preparedData != NULL);

        allocSize = this->allocBatchSize * this->chCnt * sizeof(Dtype);
        cout << "[DEBUG] allocSize = " << allocSize << endl;
        this->preparedLabel = (Dtype*)malloc(allocSize);
        SASSERT0(this->preparedLabel != NULL);

        cout << "[DEBUG] alloc ends" << endl;
    }
}

template<typename Dtype>
void ALEInputLayer<Dtype>::fillInputData(DQNImageLearner<Dtype> *dqnImage) {
    int inputDataCount = this->in_dim.batches;

    allocInputData();

    Dtype zero = 0.0;
    for (int i = 0; i < inputDataCount; i++) {
        DQNTransition<Dtype>    *rmSlot = dqnImage->getRandomRMSlot();

        int action = rmSlot->action1;
        Dtype reward1 = rmSlot->reward1;
        DQNState<Dtype>* state1 = rmSlot->state1;
        Dtype maxQ2 = rmSlot->maxQ2;

        int preparedDataOffset = i * state1->getDataCount();
        memcpy((void*)&this->preparedData[preparedDataOffset],
            (void*)state1->data, state1->getDataSize());

        // Label데이터를 action으로 부터 생성한다.
        for (int j = 0; j < this->chCnt; j++) {
            int preparedLabelOffset = this->chCnt * i + j;

            if (j == action) {
                memcpy((void*)&this->preparedLabel[preparedLabelOffset],
                    (void*)&reward1, sizeof(Dtype));
            } else {
                memcpy((void*)&this->preparedLabel[preparedLabelOffset],
                    (void*)&zero, sizeof(Dtype));
            }
        }
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

    this->_inputData[0]->set_device_with_host_data(this->preparedData, 0,
        unitSize * batchSize);
    this->_inputData[1]->set_device_with_host_data(this->preparedLabel, 0,
        this->chCnt * batchSize);
    // output : FullyConnectedLayer. 

	Layer<Dtype>::feedforward();
}

template<typename Dtype>
void ALEInputLayer<Dtype>::initialize() {
    this->type = Layer<Dtype>::ALEInput;

    // 아래의 구조체는 초기화(init())함수에서 할당하지 않는다.
    // 왜냐하면 networkConfig의 batch size정보가 있어야 할당크기를 정할 수 있는데
    // networkConfig 정보가 채워지는 타이밍이 초기화 이전이기 때문이다.
    // 혹시 바꿀 수 있을지 논의해보자.
    this->preparedData = NULL;
    this->preparedLabel = NULL;
    this->allocBatchSize = 0;
}

template<typename Dtype>
void ALEInputLayer<Dtype>::_shape(bool recursive) {
    this->out_dim = this->in_dim;

    for (uint32_t i = 0; i < this->_outputs.size(); i++) {
		this->_inputs.push_back(this->_outputs[i]);
		this->_inputData.push_back(this->_outputData[i]);
    }

	if (this->_outputs.size() > 1) {
		this->_inputData[1]->shape({this->in_dim.batches, 1, (unsigned int)this->chCnt, 1});
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
