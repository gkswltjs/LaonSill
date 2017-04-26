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

using namespace std;

#define PRINT_EDGE_DATACOUNT 3

template<typename Dtype>
void DebugUtil<Dtype>::printIndent(FILE *fp, int indent) {
    for (int i = 0; i < indent; i++)
        fprintf(fp, " ");
}

#if 1
template<typename Dtype>
void DebugUtil<Dtype>::printData(FILE *fp, Dtype data) {
    // FIXME: 
    if (sizeof(data) == sizeof(float))
    	fprintf(fp, " %f", data);
    else
    	fprintf(fp, " %lf", data);
}
#else

template<typename Dtype>
void DebugUtil<float>::printData(FILE *fp, Dtype data) {
    fprintf(fp, " %f", data);
}

template<typename Dtype>
void DebugUtil<double>::printData(FILE *fp, Dtype data) {
    fprintf(fp, " %f", data);
}
#endif

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
void DebugUtil<Dtype>::printLayerEdgesByLayerIndex(FILE *fp, const char* title,
    LayersConfig<Dtype>* lc, int layerIndex, int indent) {
    int layerCount = lc->_layers.size();
    SASSERT((layerIndex < layerCount),
        "invalid layer index. layer index=%d, layer count=%d", layerIndex, layerCount);

    printLayerEdges(fp, title, lc->_layers[layerIndex], indent);
}

template<typename Dtype>
void DebugUtil<Dtype>::printLayerEdgesByLayerName(FILE *fp, const char* title,
    LayersConfig<Dtype>* lc, string layerName, int indent) {

    SASSERT(lc->_nameLayerMap[layerName] != NULL, "invalid layer name. layer name=%s",
        layerName.c_str());

    printLayerEdges(fp, title, lc->_nameLayerMap[layerName], indent);
}

template<typename Dtype>
void DebugUtil<Dtype>::printNetworkEdges(FILE *fp, const char* title, LayersConfig<Dtype>* lc,
    int indent) {
    printIndent(fp, indent);
    fprintf(fp, "network : %s\n", title);

    for (int i = 0; i < lc->_layers.size(); i++) {
        Layer<Dtype>* layer = lc->_layers[i];
        printLayerEdgesByLayerIndex(fp, layer->getName().c_str(), lc, i, indent + 2);
    }
}

// XXX: 
#if 0
void drawAvgOfSquaredSumGrad(Gnuplot &plot, vector<pair<int, double>> &plotData,
    LayersConfig<float>* lc, string layerName) {
    // calc squared sum
    Layer<float>* layer = (Layer<float>*)lc->_nameLayerMap[layerName];
    const float* data = layer->_outputData[0]->host_grad(); 
    int nout = layer->_outputData[0]->getCount();
    float sum = 0.0;
    for (int i = 0; i < nout; i++) {
        sum += data[i] * data[i];
    }

    if (plotData.size() > 100) {
        plotData.clear();
    }

    sum /= (float)nout;
    plotData.push_back(make_pair(plotData.size(), sum));

    char cmd[1024];
    sprintf(cmd, "plot '-' using 1:2 with lines title '%s'\n", layerName.c_str());

    plot << cmd;
    plot.send1d(plotData);
}

void drawAvgOfSquaredSumData(Gnuplot &plot, vector<pair<int, double>> &plotData,
    LayersConfig<float>* lc, string layerName) {
    // calc squared sum
    Layer<float>* layer = (Layer<float>*)lc->_nameLayerMap[layerName];
    const float* data = layer->_outputData[0]->host_data(); 
    int nout = layer->_outputData[0]->getCount();
    float sum = 0.0;
    for (int i = 0; i < nout; i++) {
        sum += data[i] * data[i];
    }

    if (plotData.size() > 100) {
        plotData.clear();
    }

    sum /= (float)nout;
    plotData.push_back(make_pair(plotData.size(), sum));

    char cmd[1024];
    sprintf(cmd, "plot '-' using 1:2 with lines title '%s'\n", layerName.c_str());

    plot << cmd;
    plot.send1d(plotData);
}


typedef struct failInfo_s {
    int index;
    float soa;
    float tf;
    float err;
} failInfo;

void compareValue(LayersConfig<float>* lc, const char* layerName,
    const char* compareFilePath, int showErrCount, float errorLimit) {

    int failCount = 0;
    failInfo *fi = (failInfo*)malloc(sizeof(failInfo) * showErrCount);
    SASSERT0(fi != NULL);

    Layer<float>* layer = lc->_nameLayerMap[layerName];
    SASSERT0(layer != NULL);

    const float* soaData = layer->_outputData[0]->host_data();
    int elemCount = layer->_outputData[0]->getCount();
    printf("elem count=%d\n", elemCount);
    int elemSize = elemCount * sizeof(float);

    int fd = open(compareFilePath, O_RDONLY);
    SASSERT0(fd != -1);

    fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) & ~O_NONBLOCK);

    float* buf = (float*)malloc(elemSize);
    SASSERT0(buf != NULL);

    int nread = read(fd, buf, elemSize);
    SASSERT0(nread == elemSize);

    printf("Compare %s layer : ", layerName);
    for (int i = 0; i < elemCount; i++) {
        float err = abs(soaData[i] - buf[i]);
        if (err > errorLimit) {
            if (failCount < showErrCount) {
                fi[failCount].soa = soaData[i];
                fi[failCount].tf = buf[i];
                fi[failCount].err = err;
                fi[failCount].index = i;
            }

            failCount++;
        }
    }

    close(fd);
    free(buf);

    if (failCount == 0) {
        printf("Pass.\n");
        free(fi);
    }
    else {
        printf("Failed(%d/%d).\n", failCount, elemCount);
        
        for (int i = 0; i < min(showErrCount, failCount); i++) {
            printf("Failed. index=%d, soooa=%f, tf=%f, error=%f\n",
                fi[i].index, fi[i].soa, fi[i].tf, fi[i].err); 
        }
        free(fi);

        exit(0);
    }
}

void swapValue(LayersConfig<float>* lc, const char* layerName,
    const char* swapFilePath) {

    Layer<float>* layer = lc->_nameLayerMap[layerName];
    SASSERT0(layer != NULL);

    float* soaData = layer->_outputData[0]->mutable_host_data();
    int elemCount = layer->_outputData[0]->getCount();

    printf("elem count=%d\n", elemCount);
    int elemSize = elemCount * sizeof(float);

    int fd = open(swapFilePath, O_RDONLY);
    SASSERT0(fd != -1);

    float* buf = (float*)malloc(elemSize);
    SASSERT0(buf != NULL);

    int nread = read(fd, buf, elemSize);
    SASSERT0(nread == elemSize);

    memcpy(soaData, buf, elemSize);

    close(fd);
    free(buf);
}
#endif

template class DebugUtil<float>;
