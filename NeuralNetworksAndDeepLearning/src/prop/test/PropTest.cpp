/**
 * @file PropTest.cpp
 * @date 2017-05-04
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "PropTest.h"
#include "common.h"
#include "PropMgmt.h"
#include "StdOutLog.h"
#include "Layer.h"

using namespace std;

bool PropTest::runSimpleLayerPropTest() {
    // (1) register layer prop
    int networkID = 1;
    int layerID = 3;

#if 0
    _ConvPropLayer *convProp = (_ConvPropLayer*)malloc(sizeof(_ConvPropLayer));
    *convProp = _ConvPropLayer();
#else
    _ConvPropLayer *convProp = new _ConvPropLayer();
#endif

    LayerProp* newProp = new LayerProp(networkID, layerID, (int)Layer<float>::Conv,
        (void*)convProp);
    PropMgmt::insertLayerProp(newProp);

    _NetworkProp *networkProp = new _NetworkProp();

    PropMgmt::insertNetworkProp(networkID, networkProp);

    // (2) set layer prop and run
    PropMgmt::update(networkID, layerID);

    STDOUT_LOG("initial filter dim strides & pads value : %d, %d\n",
        SLPROP(Conv, filterDimStrides), SLPROP(Conv, filterDimPads));
    SLPROP(Conv, filterDimStrides) = 2;
    SLPROP(Conv, filterDimPads) = 1;
    STDOUT_LOG("changed filter dim strides & pads value : %d, %d\n",
        SLPROP(Conv, filterDimStrides), SLPROP(Conv, filterDimPads));

    // (3) clean up layer prop
    PropMgmt::removeLayerProp(networkID);
    PropMgmt::removeNetworkProp(networkID);

    return true;
}

bool PropTest::runSimpleNetworkPropTest() {
    // (1) register network prop
    int networkID = 2;
    int layerID = 45;

    _ConvPropLayer *convProp = new _ConvPropLayer();

    LayerProp* newProp = new LayerProp(networkID, layerID, (int)Layer<float>::Conv,
        (void*)convProp);
    PropMgmt::insertLayerProp(newProp);

    _NetworkProp *networkProp = new _NetworkProp();

    PropMgmt::insertNetworkProp(networkID, networkProp);

    // (2) set network prop and run
    PropMgmt::update(networkID, layerID);

    STDOUT_LOG("initial batchSize value : %u\n", SNPROP(batchSize));
    SNPROP(batchSize) = 128;
    STDOUT_LOG("changed batchSize value : %u\n", SNPROP(batchSize));

    // (3) clean up layer prop
    PropMgmt::removeLayerProp(networkID);
    PropMgmt::removeNetworkProp(networkID);

    return true;
}

bool PropTest::runTest() {
    bool result = runSimpleLayerPropTest();
    if (result) {
        STDOUT_LOG("*  - simple layer prop test is success");
    } else {
        STDOUT_LOG("*  - simple layer prop test is failed");
        return false;
    }

    result = runSimpleNetworkPropTest();
    if (result) {
        STDOUT_LOG("*  - simple network prop test is success");
    } else {
        STDOUT_LOG("*  - simple network prop test is failed");
        return false;
    }
    
    return true;
}
