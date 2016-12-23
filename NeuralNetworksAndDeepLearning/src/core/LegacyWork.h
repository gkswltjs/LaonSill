/**
 * @file LegacyWork.h
 * @date 2016-12-23
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef LEGACYWORK_H
#define LEGACYWORK_H 

#include "Job.h"

template <typename Dtype>
class LegacyWork {
public: 
                    LegacyWork() {}
    virtual        ~LegacyWork() {}
    static void     buildNetwork(Job* job);
    static void     trainNetwork(Job* job);
    static void     cleanupNetwork(Job* job);

    static int      createNetwork();
};
#endif /* LEGACYWORK_H */
