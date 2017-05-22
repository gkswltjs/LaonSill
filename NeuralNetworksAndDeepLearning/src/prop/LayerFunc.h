/**
 * @file LayerFunc.h
 * @date 2017-05-22
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef LAYERFUNC_H
#define LAYERFUNC_H 

typedef bool (*CBAllocInOutTensor) (void* tensorPtr, bool isInput, int index);
typedef bool (*CBAllocLayerTensors) ();
typedef bool (*CBForward) ();
typedef bool (*CBBackward) ();
typedef bool (*CBLearn) ();

typedef struct CBLayerFunc_s {
    CBAllocInOutTensor  allocInOutTensor;
    CBAllocLayerTensors allocLayerTensors;
    CBForward           forward;
    CBBackward          backward;
    CBLearn             learn;
} CBLayerFunc;

class LayerFunc {
public: 
    LayerFunc() {}
    virtual ~LayerFunc() {}

    static void init();
    static void destroy();
    static void registerLayerFunc(int layerTypeID, CBAllocInOutTensor allocInOutTensor,
                                  CBAllocLayerTensors allocLayerTensors, CBForward forward,
                                  CBBackward backward, CBLearn learn);

    static bool allocInOutTensor(int layerTypeID, void *tensorPtr, bool isInput, int index);
    static bool allocLayerTensors(int layerTypeID);
    static bool runForward(int layerTypeID);
    static bool runBackward(int layerTypeID);
    static bool learn(int layerTypeID);
private:
    static CBLayerFunc      *layerFuncs;
};

#endif /* LAYERFUNC_H */
