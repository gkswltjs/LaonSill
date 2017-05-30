/**
 * @file Update.h
 * @date 2017-05-30
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef UPDATE_H
#define UPDATE_H 

#include "common.h"
#include "Data.h"

template<typename Dtype>
class Update {
public: 
    Update() {}
    virtual ~Update() {}

    static void updateParam(const uint32_t paramSize, const Dtype regScale,
        const Dtype learnScale, const Dtype epsilon, const Dtype decayRate,
        const Dtype beta1, const Dtype beta2, Data<Dtype>* dataHistory,
        Data<Dtype>* dataHistory2, Data<Dtype>* data, float decayedBeta1,
        float decayedBeta2);

    static void doNesterov(int size, const Dtype* dx, Dtype* v_prev, Dtype* v, Dtype* x,
        const Dtype mu, const Dtype lr);

    static void doAdagrad(int size, const Dtype* dx, Dtype* cache, Dtype* x,
        const Dtype lr, const Dtype eps);

    static void doRMSprop(int size, const Dtype* dx, Dtype* cache, Dtype* x,
        const Dtype lr, const Dtype eps, const Dtype dr);

    static void doAdam(int size, const Dtype* dx, Dtype* m, Dtype* v, Dtype* x,
        const Dtype lr, const Dtype eps, const Dtype beta1, const Dtype beta2,
        const Dtype decayedBeta1, const Dtype decayedBeta2);
};

#endif /* UPDATE_H */
