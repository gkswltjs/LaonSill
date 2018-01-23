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

    DestroyNetworkReply,
    /*
     *  [Job Elem Schema for DestroyNetworkReply]
     * +------+
     * | None |
     * +------+
     */

    RunNetwork,
    /*
     *  [Job Elem Schema for RunNetwork]
     * +-----------------------------------+
     * | NetworkID (int) | inference (int) |
     * +-----------------+-----------------+
     */

    RunNetworkMiniBatch,
    /*
     *  [Job Elem Schema for RunNetworkMiniBatch]
     * +-----------------------------------+--------------------+
     * | NetworkID (int) | inference (int) | miniBatchIdx (int) |
     * +-----------------+-----------------+--------------------+
     */

    RunNetworkReply,
    /*
     *  [Job Elem Schema for RunNetworkReply]
     * +---------------+
     * | success (int) |
     * +---------------+
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

    StartInputDataProvider,
    /*
     *  [Job Elem Schema for StartInputDataProvider]
     * +-----------------+
     * | NetworkID (int) |
     * +-----------------+
     */

    GetMeasureItemName,
    /*
     *  [Job Elem Schema for GetMeasureItemName]
     * +-----------------+
     * | NetworkID (int) |
     * +-----------------+
     */

    GetMeasureItemNameReply,
    /*
     *  [Job Elem Schema for GetMeasureItemNameReply]
     * +--------------------------+
     * | measure item count (int) |  --> N
     * +----------------------+---+------------------+----+------------------------+
     * | measure item name #0 | measure item name #1 | ...| measure item name #N-1 | 
     * |   (string)           |    (string)          |    |    (string)            |
     * +----------------------+----------------------+----+------------------------+
     */

    GetMeasures,
    /*
     *  [Job Elem Schema for GetMeasures]
     * +-----------------+-----------------+-------------+-------------+
     * | NetworkID (int) | isForward (int) | start (int) | count (int) |
     * +-----------------+---------------+-+-------------+-------------+
     */

    GetMeasuresReply,
    /*
     *  [Job Elem Schema for GetMeasuresReply]
     * +-------------+
     * | count (int) |  --> N
     * +-------------+--------+
     * | start iter num (int) |
     * +----------------------+----------+-----------------------------+
     * | completed iteration count (int) | total iteration count (int) |
     * +--------------------+------------+-------+-----+---------------+--------+
     * | measure #0 (float) | measure #1 (float) | ... | measure #N - 1 (float) |
     * +-------------+------+--------------------+-----+------------------------+
     *
     *  예를들어서 아이템 카운트가 3개, 원하는 Measure iteration의 개수가 2라면..
     *   count(N)은 6을 반환한다. 또한, 
     *  [첫번째 아이템의 0번, 두번째 아이템의 0번, 세번째 아이템의 0번, 첫번째 아이템의 1번,
     *   두번째 아이템의 1번, 세번째 아이템의 1번]의 float 배열을 반환하게 된다.
     */

    // FIXME: 아래의 2개의 job은 텔코웨어 요청으로 임시적으로 만들었습니다. 
    //        job의 이름을 포함해서 옳은 방향인지에 대해서 고민이 필요합니다.

    RunNetworkWithInputData,
    /*
     *  [Job Elem Schema for RunNetworkWithInputData]
     * +-----------------+---------------+--------------+-------------+----------------------+
     * | NetworkID (int) | channel (int) | height (int) | width (int) | coord relative (int) |
     * +-----------------+--------+------+--------------+-------------+----------------------+
     * | image data (float array) |
     * +--------------------------+
     */

    RunNetworkWithInputDataReply,
    /*
     *  [Job Elem Schema for RunNetworkWithInputDataReply]
     * +--------------------+
     * | result count (int) | 
     * +-----------+--------+---+--------------+-------------+--------------------+
     * | top (int) | left (int) | bottom (int) | right (int) | confidence (float) | }
     * +-----------+--------+---+--------------+-------------+--------------------+ }
     * | labelIndex (int) |                                                         }
     * +-----------+------------+--------------+-------------+--------------------+ } result
     * | .............                                                            | } count
     * +--------------------------------------------------------------------------+
     */

    RunObjectDetectionNetworkWithInput,
    /*
     *  [Job Elem Schema for RunObjectDetectionNetworkWithInput]
     * +-----------------+---------------+--------------+-------------+-------------------------+
     * | NetworkID (int) | channel (int) | height (int) | width (int) | base network type (int) |
     * +-----------------+--------+------+--------------+-------------+-------------------------+
     * | image data (float array) |
     * +--------------------------+
     */

    RunObjectDetectionNetworkWithInputReply,
    /*
     *  [Job Elem Schema for RunObjectDetectionNetworkWithInputReply]
     * +--------------------+
     * | result count (int) | 
     * +-----------+--------+---+--------------+-------------+--------------------+
     * | top (int) | left (int) | bottom (int) | right (int) | confidence (float) | }
     * +-----------+-------+----+--------------+-------------+--------------------+ }
     * | label index (int) |                                                         }
     * +-------------------+----+--------------+-------------+--------------------+ } result
     * | .............                                                            | } count
     * +--------------------------------------------------------------------------+
     */
    
    RunClassificationNetworkWithInput,
    /*
     *  [Job Elem Schema for RunClassificationNetworkWithInput]
     * +-----------------+---------------+--------------+-------------+-------------------------+
     * | NetworkID (int) | channel (int) | height (int) | width (int) | base network type (int) |
     * +-----------------+--------+------+--------------+-------------+-------------------------+
     * | image data (float array) |
     * +--------------------------+
     */

    RunClassificationNetworkWithInputReply,
    /*
     *  [Job Elem Schema for RunClassificationNetworkWithInputReply]
     * +-------------------+
     * | label index (int) |
     * +-------------------+
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
    std::string         getStringValue(int elemIdx);
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
