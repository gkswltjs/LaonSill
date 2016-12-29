/**
 * @file DQNTransition.h
 * @date 2016-12-09
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef DQNTRANSITION_H
#define DQNTRANSITION_H 

#include "DQNState.h"

template<typename Dtype>
class DQNTransition {
public: 
                        DQNTransition() {}
    virtual            ~DQNTransition() {}

    void fill(DQNState<Dtype>* state1, int action1, Dtype reward1, 
        DQNState<Dtype>* state2, bool term) {
        this->state1    = state1;
        this->action1   = action1;
        this->reward1   = reward1;
        this->state2    = state2;
        this->term      = term;
    }

    DQNState<Dtype>    *state1;
    int                 action1; 
    Dtype               reward1;
    DQNState<Dtype>    *state2;
    bool                term;
};
#endif /* DQNTRANSITION_H */
