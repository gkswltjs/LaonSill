/**
 * @file PhysicalPlan.h
 * @date 2017-05-04
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <vector>
#include <map>
#include <mutex>
#include <atomic>

#include "common.h"

#ifndef PHYSICALPLAN_H
#define PHYSICALPLAN_H 

class PhysicalPlan {
public: 
    PhysicalPlan() {}
    virtual ~PhysicalPlan() {}

    uint64_t ppid;                      // physical plan ID
    std::vector<uint64_t> depList;      // remain dependent ppid list (for debug)
    std::atomic<int> depCount;          // remain dependent count
    std::vector<uint64_t> notifyList;   // notify ppid list
    std::vector<uint64_t> gcList;       // remain garbage collection ppid list (for debug)
    std::atomic<int> gcCount;           // remain garbage collection count

    int networkID;
    int layerID;
    int lpID;
    int classType;          // 0: forward, 1: backward, 2: update
    int nodeID;             // reserved.. 
    int devID;              // gpu device ID. resered..

    int epochIdx;
    int miniBatchIdx;       // current mini-batch (per epoch) count

    bool genBatch;          // 특별한 physical plan. Batch당 1개의 physical plan만이 
                            // 이 값이 true
                            // 이 physical plan은 다음 batch에 대한 physical plan들을 
                            // 생성하는 역할을 담당한다.

    static std::vector<uint64_t> popReadyPPIDList();
    static void markFinish(uint64_t ppID);
    static void genPPList(int networkID, int epochIdx, int miniBatchIdx);

    /*
     * physical plan 관리법
     * [developer mode]
     *   readyPPList = []
     *   while (모든 minibatch가 끝날때 까지..) {
     *      genPPList();
     *      while (true) {
     *          newPPList = popReadyPPIDList()
     *          insert new PPList into readyPPList
     *          if (newPPList is empty)
     *              break;
     *                
     *          pop PP from PPList
     *          run PP
     *          markFinish()
     *      }
     *  }
     *
     * [server-client mode]
     *     
     */

private:
    static std::atomic<uint64_t> ppIDGen;

    static std::map<uint64_t, PhysicalPlan*>     ppMap;
    static std::vector<PhysicalPlan*>            ppList;
    static std::mutex                            ppMutex;
};
#endif /* PHYSICALPLAN_H */
