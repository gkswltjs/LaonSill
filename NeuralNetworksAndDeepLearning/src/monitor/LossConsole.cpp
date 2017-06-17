/**
 * @file LossConsole.cpp
 * @date 2017-06-17
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "LossConsole.h"
#include "SysLog.h"

using namespace std;

LossConsole::LossConsole(vector<string> lossNames) {
    for (int i = 0; i < lossNames.size(); i++) {
        LossMovingAvg elem;
        elem.count = 0;
        elem.avg = 0.0;
        elem.lossName = lossNames[i];
        lossMovingAvgs.push_back(elem);
    }
}

void LossConsole::addValue(int index, float value) {
    SASSUME0(index < lossMovingAvgs.size());

    LossMovingAvg *lma = &lossMovingAvgs[index];
    lma->avg += (value - lma->avg) / (float)(lma->count + 1);
    lma->count += 1;
}

void LossConsole::printLoss(FILE* fp) {
    for (int i = 0; i < lossMovingAvgs.size(); i++) {
        fprintf(fp, "average loss[%s] : %f\n", lossMovingAvgs[i].lossName.c_str(),
            lossMovingAvgs[i].avg);
    }
}

void LossConsole::clear() {
    for (int i = 0; i < lossMovingAvgs.size(); i++) {
        lossMovingAvgs[i].count = 0;
        lossMovingAvgs[i].avg = 0.0;
    }
}
