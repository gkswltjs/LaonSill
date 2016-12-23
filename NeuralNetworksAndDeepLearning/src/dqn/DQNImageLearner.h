/**
 * @file DQNImageLearner.h
 * @date 2016-12-22
 * @author moonhoen lee
 * @brief 이미지를 입력으로 받는 Deep Q network 강화학습 모듈
 * @details
 */

#ifndef DQN_H
#define DQN_H 

#include <atomic>
#include <mutex>
#include <map>

#include "DQNTransition.h"
#include "DQNState.h"

template<typename Dtype>
class DQNImageLearner {
public: 
    DQNImageLearner(int rowCnt, int colCnt, int chCnt);
    virtual ~DQNImageLearner();

    int                     rowCnt;     // scaled row count of ALE screen
    int                     colCnt;     // scaled column count of ALE screen
    int                     chCnt;      // channel count of ALE screen
    void                    fillRM(Dtype lastReward, int lastAction, bool lastTerm,
                                Dtype* state);
    DQNTransition<Dtype>   *getRandomRMSlot();

    static void             init();
    int                     getID() { return this->dqnID; }

    static std::map<int, DQNImageLearner<Dtype>*>   learnerIDMap;
    static std::mutex       learnerIDMapMutex;

    static DQNImageLearner<Dtype>*  getLearnerFromID(int dqnID);
private:
    DQNTransition<Dtype>  **rmSlots;    // replay memory slots
    DQNState<Dtype>       **stateSlots; // saved state slots for replay memory slots

    int                     rmSlotCnt;      // element count of replay memory
    int                     stateSlotCnt;   // element count of replay memory

    int                     rmSlotHead; // circular queue head for replay memory slots
    int                     stateSlotHead;  // circular queue head for saved state slots

    DQNState<Dtype>        *lastState;

    int                     dqnID;
    static std::atomic<int> dqnIDGen;
};
#endif /* DQN_H */
