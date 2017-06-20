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
#include <vector>

/*
 * Job Description
 * +------------+--------------+-----------------+------------------------------+
 * | JobID(int) | JobType(int) | JobElemCnt(int) | JobElemTypes(int*JobElemCnt) |
 * +----------------------------------------------------------------------------+
 * | JobElems(variable) |
 * +--------------------+
 */

typedef enum JobType_e {
    HaltMachine,
    /*
     *  [Job Elem Schema for HaltMachine]
     * +------+
     * | None |
     * +------+
     */


    TestJob,
    /*
     *  [Job Elem Schema for TestJob]
     * +------------------------------------------------------+
     * | A (int) | B (float) | C (float array, 100) | D (int) |
     * +------------------------------------------------------+
     */

    CreateNetworkFromFile,
    /*
     *  [Job Elem Schema for CreateNetworkFromFile]
     * +----------------------+
     * | JSONFilePath(string) |
     * +----------------------+
     */

    CreateNetwork,
    /*
     *  [Job Elem Schema for CreateNetwork]
     * +---------------------------+
     * | NetworkDefinition(string) |
     * +---------------------------+
     */

    CreateNetworkReply,
    /*
     *  [Job Elem Schema for CreateNetworkReply]
     * +----------------+
     * | NetworkID(int) |
     * +----------------+
     */

    DestroyNetwork,
    /*
     *  [Job Elem Schema for DestroyNetwork]
     * +-----------------+
     * | NetworkID (int) |
     * +-----------------+
     */

    RunNetwork,
    /*
     *  [Job Elem Schema for RunNetwork]
     * +-----------------------------------+
     * | NetworkID (int) | inference (int) |
     * +-----------------+-----------------+
     */

    RunNetworkReply,
    /*
     *  [Job Elem Schema for RunNetworkReply]
     * +------+
     * | None |
     * +------+
     */

    BuildNetwork,
    /*
     *  [Job Elem Schema for BuildNetwork]
     * +--------------------------------+
     * | NetworkID (int) | epochs (int) |
     * +-----------------+--------------+
     */

    BuildNetworkReply,
    /*
     *  [Job Elem Schema for BuildNetworkReply]
     * +------+
     * | None |
     * +------+
     */

    ResetNetwork,
    /*
     *  [Job Elem Schema for ResetNetwork]
     * +-----------------+
     * | NetworkID (int) |
     * +-----------------+
     */

    ResetNetworkReply,
    /*
     *  [Job Elem Schema for ResetNetworkReply]
     * +------+
     * | None |
     * +------+
     */

    RunNetworkMiniBatch,
    /*
     *  [Job Elem Schema for RunNetworkMiniBatch]
     * +-----------------------------------+--------------------+
     * | NetworkID (int) | inference (int) | miniBatchIdx (int) |
     * +-----------------+-----------------+--------------------+
     */

    RunNetworkMiniBatchReply,
    /*
     *  [Job Elem Schema for RunNetworkMiniBatchReply]
     * +------+
     * | None |
     * +------+
     */

    SaveNetwork,
    /*
     *  [Job Elem Schema for SaveNetwork]
     * +------------------------------------+
     * | NetworkID (int) | filePath(string) |
     * +-----------------+------------------+
     */

    SaveNetworkReply,
    /*
     *  [Job Elem Schema for SaveNetworkReply]
     * +------+
     * | None |
     * +------+
     */
    
    LoadNetwork,
    /*
     *  [Job Elem Schema for LoadNetwork]
     * +------------------------------------+
     * | NetworkID (int) | filePath(string) |
     * +-----------------+------------------+
     */

    LoadNetworkReply,
    /*
     *  [Job Elem Schema for LoadNetworkReply]
     * +------+
     * | None |
     * +------+
     */
    
    JobTypeMax

} JobType;

class Job {
public:
    enum JobElemType : int {
        IntType = 0,                // int
        FloatType,              // float
        FloatArrayType,         // length(int) + (float * length)
        StringType,             // length(int) + (char * length)
        ElemTypeMax
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
    const char*         getStringValue(int elemIdx);
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

    static std::vector<JobType>     pubJobTypeMap;
};

#endif /* JOB_H */
