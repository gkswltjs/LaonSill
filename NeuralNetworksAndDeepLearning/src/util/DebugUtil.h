/**
 * @file DebugUtil.h
 * @date 2017-03-23
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef DEBUGUTIL_H
#define DEBUGUTIL_H 

#include <string>

#include "common.h"
#include "NetworkConfig.h"
#include "Layer.h"

template<typename Dtype>
class DebugUtil {
public: 
    // exclusive...
    enum PrintDataType : int {
        PrintData = 1,
        PrintGrad = 2
    };

    DebugUtil() {}
    virtual ~DebugUtil() {}

    static void printIndent(FILE *fp, int indent);
    static void printData(FILE *fp, Dtype data);
    static void printEdges(FILE *fp, const char* title, Data<Dtype>* data, int flags,
        int indent);
    static void printLayerEdges(FILE *fp, const char* title, Layer<Dtype>* layer, int indent);
#if 0
    static void printLayerEdgesByLayerIndex(FILE *fp, const char* title,
        LayersConfig<Dtype>* lc, int layerIndex, int indent);
    static void printLayerEdgesByLayerName(FILE *fp, const char* title,
        LayersConfig<Dtype>* lc, std::string layerName, int indent);
    static void printNetworkEdges(FILE *fp, const char* title, LayersConfig<Dtype>* lc,
        int indent);
#endif
};
#endif /* DEBUGUTIL_H */
