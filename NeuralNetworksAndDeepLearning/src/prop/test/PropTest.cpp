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

bool PropTest::runSimplePropTest() {
    // (1) register layer prop
    int networkID = 1;
    int layerID = 3;

    _ConvPropLayer *convProp = (_ConvPropLayer*)malloc(sizeof(_ConvPropLayer));
    *convProp = _ConvPropLayer();

    LayerProp* newProp = new LayerProp(networkID, layerID, (int)Layer<float>::Conv,
        (void*)convProp);
    PropMgmt::insertLayerProp(newProp);

    // (2) set layer prop and run
    PropMgmt::update(networkID, layerID);

    STDOUT_LOG("initial filter dim strides & pads value : %d, %d\n",
        SPROP(Conv, filterDimStrides), SPROP(Conv, filterDimPads));
    SPROP(Conv, filterDimStrides) = 2;
    SPROP(Conv, filterDimPads) = 1;
    STDOUT_LOG("changed filter dim strides & pads value : %d, %d\n",
        SPROP(Conv, filterDimStrides), SPROP(Conv, filterDimPads));

    // (3) clean up layer prop
    PropMgmt::removeLayerProp(networkID);

    return true;
}

bool PropTest::runTest() {
    bool result = runSimplePropTest();
    if (result) {
        STDOUT_LOG("*  - simple prop test is success");
    } else {
        STDOUT_LOG("*  - simple prop test is failed");
        return false;
    }
    
    return true;
}
