/**
 * @file LayerFunc.h
 * @date 2017-05-22
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef LAYERFUNC_H
#define LAYERFUNC_H 

typedef void* (*CBInitLayer) ();
typedef void (*CBDestroyLayer) (void* instancePtr);
typedef void (*CBSetInOutTensor) (void* instancePtr, void* tensorPtr, bool isInput, int index);
typedef bool (*CBAllocLayerTensors) (void* instancePtr);
typedef void (*CBForward) (void* instancePtr, int miniBatchIndex);
typedef void (*CBBackward) (void* instancePtr);
typedef void (*CBLearn) (void* instancePtr);

typedef struct CBLayerFunc_s {
    CBInitLayer         initLayer;
    CBDestroyLayer      destroyLayer;
    CBSetInOutTensor    setInOutTensor;
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
    static void registerLayerFunc(int layerType, CBInitLayer initLayer,
                                  CBDestroyLayer destroyLayer,
                                  CBSetInOutTensor setInOutTensor, 
                                  CBAllocLayerTensors allocLayerTensors, CBForward forward,
                                  CBBackward backward, CBLearn learn);

    static void* initLayer(int layerType);
    static void destroyLayer(int layerType, void* instancePtr);
    static void setInOutTensor(int layerType, void* instancePtr, void *tensorPtr,
        bool isInput, int index);
    static bool allocLayerTensors(int layerType, void* instancePtr);
    static void runForward(int layerType, void* instancePtr, int miniBatchIdx);
    static void runBackward(int layerType, void* instancePtr);
    static void learn(int layerType, void* instancePtr);
private:
    static CBLayerFunc      *layerFuncs;
};

#endif /* LAYERFUNC_H */
