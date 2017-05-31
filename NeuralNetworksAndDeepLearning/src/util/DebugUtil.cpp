/**
 * @file DebugUtil.cpp
 * @date 2017-03-23
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "DebugUtil.h"
#include "SysLog.h"

#include "FullyConnectedLayer.h"
#include "ConvLayer.h"
#include "WorkContext.h"
#include "PropMgmt.h"

using namespace std;

#define PRINT_EDGE_DATACOUNT 3

template<typename Dtype>
void DebugUtil<Dtype>::printIndent(FILE *fp, int indent) {
    for (int i = 0; i < indent; i++)
        fprintf(fp, " ");
}

template<typename Dtype>
void DebugUtil<Dtype>::printData(FILE *fp, Dtype data) {
    // FIXME: 
    if (sizeof(data) == sizeof(float))
    	fprintf(fp, " %f", data);
    else
    	fprintf(fp, " %lf", data);
}

template<typename Dtype>
void DebugUtil<Dtype>::printEdges(FILE *fp, const char* title, Data<Dtype>* data, int flags,
    int indent) {
    int dataCount = data->getCount();
    const Dtype* hostData = data->host_data();
    const Dtype* hostGrad = data->host_grad();

    SASSERT(((flags & DebugUtil<Dtype>::PrintData) || (flags & DebugUtil<Dtype>::PrintGrad)),
        "invalid flag option. flags=%d", flags);

    if (flags & DebugUtil<Dtype>::PrintData) {
        printIndent(fp, indent);
        fprintf(fp, "- %s(data) : ", title);
        for (int i = 0; i < min(PRINT_EDGE_DATACOUNT, dataCount); i++) {
            printData(fp, hostData[i]);
        }
        fprintf(fp, " ~ ");
        for (int i = max(0, dataCount - PRINT_EDGE_DATACOUNT); i < dataCount; i++) {
            printData(fp, hostData[i]);
        }
        fprintf(fp, "\n");
    }

    if (flags & DebugUtil<Dtype>::PrintGrad) {
        printIndent(fp, indent);
        fprintf(fp, "- %s(grad) : ", title);
        for (int i = 0; i < min(PRINT_EDGE_DATACOUNT, dataCount); i++) {
            printData(fp, hostGrad[i]);
        }
        fprintf(fp, " ~ ");
        for (int i = max(0, dataCount - PRINT_EDGE_DATACOUNT); i < dataCount; i++) {
            printData(fp, hostGrad[i]);
        }
        fprintf(fp, "\n");
    }
}

template<typename Dtype>
void DebugUtil<Dtype>::printLayerEdges(FILE *fp, const char* title, Layer<Dtype>* layer,
    int indent) {
    printIndent(fp, indent);
    fprintf(fp, "layer : %s\n", title);

    int flags = (DebugUtil<Dtype>::PrintData | DebugUtil<Dtype>::PrintGrad);

    FullyConnectedLayer<Dtype>* fcLayer = 
        dynamic_cast<FullyConnectedLayer<Dtype>*>(layer);
    if (fcLayer) {
        printEdges(fp, "weight", fcLayer->_params[0], flags, indent + 2);
        printEdges(fp, "bias", fcLayer->_params[1], flags, indent + 2);
    }

    ConvLayer<Dtype>* convLayer = dynamic_cast<ConvLayer<Dtype>*>(layer);
    if (convLayer) {
        printEdges(fp, "filter", convLayer->_params[0], flags, indent + 2);
        printEdges(fp, "bias", convLayer->_params[1], flags, indent + 2);
    }

    Data<Dtype>* inputData = layer->_inputData[0];
    Data<Dtype>* outputData = layer->_outputData[0];

    char layerNameBuf[32];
    for (int i = 0; i < layer->_inputData.size(); i++) {
        sprintf(layerNameBuf, "input[%d]", i);
        printEdges(fp, layerNameBuf, layer->_inputData[i], flags, indent + 2);
    }
    for (int i = 0; i < layer->_outputData.size(); i++) {
        sprintf(layerNameBuf, "output[%d]", i);
        printEdges(fp, layerNameBuf, layer->_outputData[i], flags, indent + 2);
    }
}

template<typename Dtype>
void DebugUtil<Dtype>::printLayerEdgesByLayerID(FILE *fp, const char* title,
    int networkID, int layerID, int indent) {
    int oldNetworkID = WorkContext::curNetworkID;
    WorkContext::updateNetwork(networkID);
    PhysicalPlan* pp = WorkContext::curPhysicalPlan;

    SASSERT(pp->instanceMap.find(layerID) != pp->instanceMap.end(),
        "invalid layer ID. layer ID=%d.", layerID);

    Layer<Dtype>* layer = (Layer<Dtype>*)pp->instanceMap[layerID];
    WorkContext::updateLayer(networkID, layerID);
    printLayerEdges(fp, SLPROP_BASE(name).c_str(), layer, indent + 2);

    WorkContext::updateNetwork(oldNetworkID);
}

template<typename Dtype>
void DebugUtil<Dtype>::printLayerEdgesByLayerName(FILE *fp, const char* title,
    int networkID, string layerName, int indent) {

    int oldNetworkID = WorkContext::curNetworkID;
    WorkContext::updateNetwork(networkID);
    PhysicalPlan* pp = WorkContext::curPhysicalPlan;

    Layer<Dtype>* layer;
    bool foundLayer = false;
    for (map<int, void*>::iterator iter = pp->instanceMap.begin();
        iter != pp->instanceMap.end(); iter++) {
        int layerID = iter->first;
        void* instancePtr = iter->second;

        WorkContext::updateLayer(networkID, layerID);

        layer = (Layer<Dtype>*)instancePtr;

        // FIXME: 현재 linear search. 너무 속도가 느리면 개선하자.
        if (SLPROP_BASE(name) == layerName) {
            printLayerEdges(fp, SLPROP_BASE(name).c_str(), layer, indent + 2);
            foundLayer = true;
            break;
        }
    }

    SASSERT(foundLayer, "invalid layer name. layer name=%s", layerName.c_str());
    WorkContext::updateNetwork(oldNetworkID);
}

template<typename Dtype>
void DebugUtil<Dtype>::printNetworkEdges(FILE *fp, const char* title, int networkID,
    int indent) {
    printIndent(fp, indent);
    fprintf(fp, "network : %s\n", title);

    int oldNetworkID = WorkContext::curNetworkID;
    WorkContext::updateNetwork(networkID);
    PhysicalPlan* pp = WorkContext::curPhysicalPlan;

    for (map<int, void*>::iterator iter = pp->instanceMap.begin();
        iter != pp->instanceMap.end(); iter++) {
        int layerID = iter->first;
        void* instancePtr = iter->second;

        WorkContext::updateLayer(networkID, layerID);

        Layer<Dtype>* layer = (Layer<Dtype>*)instancePtr;
        printLayerEdges(fp, SLPROP_BASE(name).c_str(), layer, indent + 2);
    }

    WorkContext::updateNetwork(oldNetworkID);
}

template class DebugUtil<float>;
