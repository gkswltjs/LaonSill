/**
 * @file DQNImageLearner.cpp
 * @date 2016-12-22
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <stdlib.h>
#include <time.h>

#include <cstdlib>

#include "common.h"
#include "DQNImageLearner.h"

using namespace std;

template<typename Dtype>
atomic<int> DQNImageLearner<Dtype>::dqnIDGen;
template<typename Dtype>
map<int, DQNImageLearner<Dtype>*> DQNImageLearner<Dtype>::learnerIDMap;
template<typename Dtype>
mutex DQNImageLearner<Dtype>::learnerIDMapMutex;

template<typename Dtype>
DQNImageLearner<Dtype>::DQNImageLearner(int rowCnt, int colCnt, int chCnt, int actionCnt) {
    this->rowCnt    = rowCnt;
    this->colCnt    = colCnt;
    this->chCnt     = chCnt;
    this->actionCnt = actionCnt;
    this->rmSlotCnt = SPARAM(DQN_REPLAY_MEMORY_ELEM_COUNT);
    this->rmReadyCountDown = this->rmSlotCnt + 2;

    this->epsilon   = SPARAM(DQN_DEFAULT_EPSILON_VALUE);
    this->gamma     = SPARAM(DQN_DEFAULT_GAMMA_VALUE);

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

    this->lastState = NULL;

    this->dqnID = atomic_fetch_add(&DQNImageLearner<Dtype>::dqnIDGen, 1);

    unique_lock<mutex> learnerLock(DQNImageLearner<Dtype>::learnerIDMapMutex);
    DQNImageLearner<Dtype>::learnerIDMap[this->dqnID] = this;
    learnerLock.unlock();

    int maxQHeadValueAllocSize = SPARAM(DQN_DEFAULT_MINI_BATCH_COUNT) * sizeof(Dtype);
    this->maxQHeadValues = (Dtype*)malloc(maxQHeadValueAllocSize);
    SASSERT0(this->maxQHeadValues != NULL);

    int qLabelValueAllocSize = SPARAM(DQN_DEFAULT_MINI_BATCH_COUNT) * sizeof(Dtype);
    this->qLabelValues = (Dtype*)malloc(qLabelValueAllocSize);
    SASSERT0(this->qLabelValues != NULL);

    int activeRMSlotAllocSize =
        SPARAM(DQN_DEFAULT_MINI_BATCH_COUNT) * sizeof(DQNTransition<Dtype>*);
    this->activeRMSlots = (DQNTransition<Dtype>**)malloc(activeRMSlotAllocSize);
    SASSERT0(this->activeRMSlots != NULL);

    for (int i = 0; i < SPARAM(DQN_DEFAULT_MINI_BATCH_COUNT); i++) {
        this->activeRMSlots[i] = new DQNTransition<Dtype>();     
        SASSERT0(this->activeRMSlots[i] != NULL);
    }

    this->learningCnt = 0;
}

template<typename Dtype>
DQNImageLearner<Dtype>::~DQNImageLearner() {
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

    unique_lock<mutex> learnerLock(DQNImageLearner<Dtype>::learnerIDMapMutex);
    DQNImageLearner<Dtype>::learnerIDMap.erase(this->dqnID);
    learnerLock.unlock();

    SASSERT0(this->maxQHeadValues != NULL);
    free(this->maxQHeadValues);

    SASSERT0(this->qLabelValues != NULL);
    free(this->qLabelValues);

    SASSERT0(this->activeRMSlots != NULL);
    for (int i = 0; i < SPARAM(DQN_DEFAULT_MINI_BATCH_COUNT); i++) {
        delete this->activeRMSlots[i];
    }
    free(this->activeRMSlots);
}

template<typename Dtype>
void DQNImageLearner<Dtype>::fillRM(Dtype lastReward, int lastAction, bool lastTerm,
    Dtype* state) {
    int copySize = this->stateSlots[this->stateSlotHead]->getDataSize();
    memcpy((void*)this->stateSlots[this->stateSlotHead]->data, (void*)state, copySize);

    if (this->lastState != NULL) {
        this->rmSlots[this->rmSlotHead]->fill(this->stateSlots[this->rmSlotHead],
            lastAction, lastReward, this->lastState, lastTerm);

        // active RM Slots의 첫번째 원소는 가장 최신에 삽입된 RM이다.
        // 나중에 feed forawrd를 할때에 이 데이터를 활용.
        this->activeRMSlots[0] = this->rmSlots[this->rmSlotHead];
        this->rmSlotHead = (this->rmSlotHead + 1) % this->rmSlotCnt;
    }

    this->lastState = this->stateSlots[this->stateSlotHead];
    this->stateSlotHead = (this->stateSlotHead + 1) % this->stateSlotCnt;

    if (this->rmReadyCountDown != 0)
        this->rmReadyCountDown -= 1;

}

template<typename Dtype>
DQNTransition<Dtype>* DQNImageLearner<Dtype>::getRandomRMSlot() {
    srand(time(NULL));

    int index = rand() % this->rmSlotCnt;
    return this->rmSlots[index];
}

template<typename Dtype>
void DQNImageLearner<Dtype>::init() {
    atomic_store(&DQNImageLearner<Dtype>::dqnIDGen, 0);
}

template<typename Dtype>
DQNImageLearner<Dtype>* DQNImageLearner<Dtype>::getLearnerFromID(int dqnID) {
    DQNImageLearner<Dtype>* learner;

    unique_lock<mutex> learnerLock(DQNImageLearner<Dtype>::learnerIDMapMutex);
    learner = DQNImageLearner<Dtype>::learnerIDMap[dqnID];
    learnerLock.unlock();

    return learner;
}

template<typename Dtype>
int DQNImageLearner<Dtype>::chooseAction(Network<Dtype>* netQ) {
    bool useRandomPolicy;

    // (1) exploration vs exploitation by e-soft algorithm
    srand(time(NULL));
    float randomValue = (float)rand() / (float)RAND_MAX;

    if (this->epsilon < randomValue) {
        useRandomPolicy = false;
    } else {
        useRandomPolicy = true;
    }

    if (!isReady())
        useRandomPolicy = true;

    // (2) choose action
    int action;
    if (useRandomPolicy) {
        action = rand() % this->actionCnt;
    } else {
        vector<Data<Dtype>*> outputData;

        outputData = netQ->feedForwardDQNNetwork(1, this, true);

        action = 0;
        const Dtype *output = outputData[0]->host_data();
        Dtype maxQVal = output[0];
        for (int i = 1; i < this->actionCnt; i++) {
            if (maxQVal < output[i]) {
                action = i;
                maxQVal = output[i];
            }
        }
    }

    return action;
}

template<typename Dtype>
bool DQNImageLearner<Dtype>::isReady() {
    if (this->rmReadyCountDown == 0)
        return true;
    else
        return false;
}

template<typename Dtype>
void DQNImageLearner<Dtype>::prepareActiveRMSlots() {
    for (int i = 0; i < SPARAM(DQN_DEFAULT_MINI_BATCH_COUNT); i++)
        this->activeRMSlots[i] = this->getRandomRMSlot();
}

template<typename Dtype>
void DQNImageLearner<Dtype>::forwardMiniBatch(Network<Dtype>* network, bool isNetQ) {
    int miniBatchCount = SPARAM(DQN_DEFAULT_MINI_BATCH_COUNT);
    vector<Data<Dtype>*> outputData;

    outputData = network->feedForwardDQNNetwork(miniBatchCount, this, isNetQ);

    if (!isNetQ) {
        const Dtype *output = outputData[0]->host_data();

        for (int i = 0; i < miniBatchCount; i++) {
            int baseIndex = i * this->actionCnt;
            Dtype maxQVal = output[baseIndex];
            for (int j = 1; j < this->actionCnt; j++) {
                int index = baseIndex + j;
                if (maxQVal < output[index]) {
                    maxQVal = output[index];
                }
            }

            this->maxQHeadValues[i] = maxQVal;
        }
    }
}

template<typename Dtype>
void DQNImageLearner<Dtype>::updateQLabelValues() {
    for (int i = 0; i < SPARAM(DQN_DEFAULT_MINI_BATCH_COUNT); i++) {
        DQNTransition<Dtype> *rmSlot = this->activeRMSlots[i];
        Dtype maxQHeadValue = this->maxQHeadValues[i];

        if (rmSlot->term == 1) {
            this->qLabelValues[i] = rmSlot->reward1;
        } else {
            this->qLabelValues[i] = rmSlot->reward1 + (Dtype)this->gamma * maxQHeadValue;
        }
    }
}

template<typename Dtype>
void DQNImageLearner<Dtype>::backwardMiniBatch(Network<Dtype>* network) {
    network->backPropagateDQNNetwork(this);
}

template<typename Dtype>
void DQNImageLearner<Dtype>::syncNetworks(Network<Dtype>* netQ, Network<Dtype>* netQHead) {
    this->learningCnt++;

    if (this->learningCnt % SPARAM(DQN_DEFAULT_SYNC_NETWORKS_PERIOD) == 0) {
        netQ->syncNetwork(netQHead);
    }
}

template class DQNImageLearner<float>;
