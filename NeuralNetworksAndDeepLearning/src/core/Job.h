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
#include <map>
#include <mutex>

/*
 * Job Description
 * +------------+--------------+-----------------+------------------------------+
 * | JobID(int) | JobType(int) | JobElemCnt(int) | JobElemTypes(int*JobElemCnt) |
 * +----------------------------------------------------------------------------+
 * | JobElems(variable) |
 * +--------------------+
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
         *  [Job Elem Schema for BuildNetwork]
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
         *  [Job Elem Schema for CleanupNetwork]
         * +------------------+
         * | network Id (int) |
         * +------------------+
         */

        CreateDQNImageLearner,
        /*
         *  [Job Elem Schema for CreateDQNImage]
         * +-----------------+--------------------+---------------------+
         * | row count (int) | column count (int) | channel count (int) |
         * +-----------------+--------------------+---------------------+
         */

        CreateDQNImageLearnerResult,
        /*
         *  [Job Elem Schema for CreateDQNImageResult]
         * +----------------------------+--------------------+-------------------------+
         * | DQN Image learner ID (int) | network Q ID (int) | network Q head ID (int) |
         * +----------------------------+--------------------+-------------------------+
         */

        // DQN related jobs
        BuildDQNNetworks,
        /*
         *  [Job Elem Schema for BuildDQNNetwork]
         * +----------------------------+--------------------+-------------------------+
         * | DQN Image learner ID (int) | network Q ID (int) | network Q head ID (int) |
         * +----------------------------+--------------------+-------------------------+
         */

        PushDQNImageInput,
        /*
         *  [Job Elem Schema for PushDQNImageInput]
         * +--------------------------------------------------------------------+
         * | DQN Image learner Id (int) | reward t-1 (float) | action t-1 (int) |
         * |--------------------------------------------------------------------+
         * | term t-1 (int) | state t (float array, 4 * 84 * 84) |
         * +-----------------------------------------------------+
         */

        FeedForwardDQNNetwork,

        HaltMachine,


        TestJob,
        /*
         *  [Job Elem Schema for TestJob]
         * +------------------------------------------------------+
         * | A (int) | B (float) | C (float array, 100) | D (int) |
         * +------------------------------------------------------+
         */
        JobTypeMax
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

    static int          genJobID();
    int                 genTaskID();

    int                 getJobID();
    static void         init();

    bool                hasPubJob();
    JobType             getPubJobType();

    static std::map<int, Job*>      reqPubJobMap;
    static std::mutex   reqPubJobMapMutex;

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


    static std::atomic<int>         jobIDGen;
    std::atomic<int>                taskIDGen;

    int                             jobID;
};

#endif /* JOB_H */
