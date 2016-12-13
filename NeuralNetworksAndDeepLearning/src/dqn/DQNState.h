/**
 * @file DQNState.h
 * @date 2016-12-09
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef DQNSTATE_H
#define DQNSTATE_H 

#include <stdlib.h>

#include "SysLog.h"

template <typename Dtype>
class DQNState {
public: 
    DQNState(int rows, int cols, int channels) {
        this->rows = rows;
        this->cols = cols;
        this->channels = channels;

        // FIXME: 이렇게 DQNState마다 메모리를 할당해서 쓰면 비효율적이다.
        //      하지만 처음 slot을 채울때에만 malloc()이 호출이 되기 때문에 큰 문제가
        //      되지는 않는다. 추후에 예쁘게 수정을 하도록 하자.
        int allocSize = sizeof(Dtype) * this->rows * this->cols * this->channels;
        this->data = (Dtype*)malloc(allocSize);
        SASSERT0(this->data != NULL);
    }

    virtual ~DQNState() {
        if (this->data != NULL)
            free(this->data);
    }

    int             rows;
    int             cols;
    int             channels;

    Dtype*          data;

    int             getDataSize() {
        return this->rows * this->cols * this->channels * sizeof(Dtype);
    }

    int             getDataCount() {
        return this->rows * this->cols * this->channels;
    }
};

#endif /* DQNSTATE_H */
