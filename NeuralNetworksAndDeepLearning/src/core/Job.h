/**
 * @file Job.h
 * @date 2016-10-14
 * @author moonhoen lee
 * @brief 서버가 수행해야 할 작업을 명시한다.
 * @details
 */

#ifndef JOB_H
#define JOB_H 

#include <atomic>

/*
 * Job Description
 * +--------------+-----------------+------------------------------+--------------------+
 * | JobType(int) | JobElemCnt(int) | JobElemTypes(int*JobElemCnt) | JobElems(variable) |
 * +--------------+-----------------+------------------------------+--------------------+
 */

class Job {
public:
    enum JobElemType : int {
        IntType = 0,                // int
        FloatType,              // float
        FloatArrayType,         // length(int) + (float * length)
        ElemTypeMax
    };

    enum JobType : int {
        // belows will be deprecated
        BuildNetwork = 0,
        /*
         *  [Job Elem Schema for BuildLayer]
         * +------------------+
         * | network Id (int) |
         * +------------------+
         */
        TrainNetwork,
        /*
         *  [Job Elem Schema for TrainNetwork]
         * +------------------+-------------------+
         * | network Id (int) | batch count (int) |
         * +------------------+-------------------+
         */
        CleanupNetwork,
        /*
         *  [Job Elem Schema for CleanupLayer]
         * +------------------+
         * | network Id (int) |
         * +------------------+
         */

        // DQN related jobs
        BuildDQNNetwork,
        /*
         *  [Job Elem Schema for CleanupLayer]
         * +------------------+
         * | network Id (int) |
         * +------------------+
         */

        PushDQNInput,
        /*
         *  [Job Elem Schema for CleanupLayer]
         * +------------------+------------------------+
         * | network Id (int) | ....
         * +------------------+
         */

        FeedForwardDQNNetwork,

        HaltMachine,
        TypeMax
    };

    typedef struct JobElemDef_s {
        JobElemType     elemType;
        int             elemOffset;
        int             arrayCount;
    } JobElemDef;

                        Job(JobType jobType, int jobElemCnt, JobElemType *jobElemTypes,
                            char *jobElemValues);

                        Job(JobType jobType);   // for incremental build

    virtual            ~Job();

    // for incremental build
    void                addJobElem(JobElemType jobElemType, int arrayCount, void* dataPtr);    

    JobType             getType() const { return this->jobType; }
    int                 getJobElemCount() const { return this->jobElemCnt; }
    int                 getIntValue(int elemIdx);
    float               getFloatValue(int elemIdx);
    float              *getFloatArray(int elemIdx);
    float               getFloatArrayValue(int elemIdx, int arrayIdx);
    JobElemDef          getJobElemDef(int elemIdx);

    int                 getJobSize();

    std::atomic<int>    refCnt;     // for multiple consumer

private:
    JobType             jobType;
    int                 jobElemCnt;
    JobElemDef         *jobElemDefs;
    char               *jobElemValues;

    int                 getJobElemValueSize();
    bool                isVaildElemIdx(int elemIdx);
    bool                isValidElemValue(JobElemType elemType, int elemIdx);
    bool                isValidElemArrayValue(JobElemType elemType, int elemIdx,
                            int arrayIdx);
};

#endif /* JOB_H */
