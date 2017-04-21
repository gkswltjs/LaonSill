/**
 * @file YOLO.cpp
 * @date 2017-04-20
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <vector>

#include "common.h"
#include "Debug.h"
#include "YOLO.h"
#include "StdOutLog.h"
#include "NetworkConfig.h"
#include "NetworkMonitor.h"
#include "ImageUtil.h"
#include "DebugUtil.h"

using namespace std;

template<typename Dtype>
LayersConfig<Dtype>* YOLO<Dtype>::createYoloPreLayersConfig() {
    const float bias_const = 0.2f;

    LayersConfig<Dtype>* layersConfig = (new typename LayersConfig<Dtype>::Builder())

        ->layer((new typename ILSVRCInputLayer<Dtype>::Builder())
            ->id(0)
            ->name("ILSVRCInputLayer")
            ->imageDir(std::string(SPARAM(BASE_DATA_DIR))
                + std::string("/ilsvrc12_train/"))
            ->source(std::string(SPARAM(BASE_DATA_DIR))
                + std::string("/ilsvrc12_train/"))
            ->resizeImage(448, 448)
            ->sourceType("ImagePack")       // dummy
            ->outputs({"data", "label"}))

        // 1st group
        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(1001)
                ->name("conv1_1")
                ->filterDim(7, 7, 3, 64, 1, 2)
                ->weightUpdateParam(1, 1)
                ->biasUpdateParam(2, 0)
                ->weightFiller(ParamFillerType::Xavier, 0.1)
                ->biasFiller(ParamFillerType::Constant, bias_const)
                ->inputs({"data"})
                ->outputs({"conv1_1"}))

        ->layer((new typename DropOutLayer<Dtype>::Builder())
                ->id(1002)
                ->probability(0.5)
                ->name("dropout1_1")
                ->inputs({"conv1_1"})
                ->outputs({"conv1_1"}))

        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(1003)
                ->leaky(0.1)
                ->name("relu1_1")
                ->inputs({"conv1_1"})
                ->outputs({"conv1_1"}))

        ->layer((new typename PoolingLayer<Dtype>::Builder())
                ->id(1004)
                ->name("pool1_1")
                ->poolDim(2, 2, 0, 2)
                ->poolingType(Pooling<Dtype>::Max)
                ->inputs({"conv1_1"})
                ->outputs({"pool1_1"}))


        // 2nd group
        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(2001)
                ->name("conv2_1")
                ->filterDim(3, 3, 64, 192, 1, 1)
                ->weightUpdateParam(1, 1)
                ->biasUpdateParam(2, 0)
                ->weightFiller(ParamFillerType::Xavier, 0.1)
                ->biasFiller(ParamFillerType::Constant, bias_const)
                ->inputs({"pool1_1"})
                ->outputs({"conv2_1"}))

        ->layer((new typename DropOutLayer<Dtype>::Builder())
                ->id(2002)
                ->probability(0.5)
                ->name("dropout2_1")
                ->inputs({"conv2_1"})
                ->outputs({"conv2_1"}))

        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(2003)
                ->leaky(0.1)
                ->name("relu2_1")
                ->inputs({"conv2_1"})
                ->outputs({"conv2_1"}))

        ->layer((new typename PoolingLayer<Dtype>::Builder())
                ->id(2004)
                ->name("pool2_1")
                ->poolDim(2, 2, 0, 2)
                ->poolingType(Pooling<Dtype>::Max)
                ->inputs({"conv2_1"})
                ->outputs({"pool2_1"}))



        // 3rd group
        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(3001)
                ->name("conv3_1")
                ->filterDim(1, 1, 192, 128, 0, 1)
                ->weightUpdateParam(1, 1)
                ->biasUpdateParam(2, 0)
                ->weightFiller(ParamFillerType::Xavier, 0.1)
                ->biasFiller(ParamFillerType::Constant, bias_const)
                ->inputs({"pool2_1"})
                ->outputs({"conv3_1"}))

        ->layer((new typename DropOutLayer<Dtype>::Builder())
                ->id(3002)
                ->probability(0.5)
                ->name("dropout3_1")
                ->inputs({"conv3_1"})
                ->outputs({"conv3_1"}))

        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(3003)
                ->leaky(0.1)
                ->name("relu3_1")
                ->inputs({"conv3_1"})
                ->outputs({"conv3_1"}))

        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(3004)
                ->name("conv3_2")
                ->filterDim(3, 3, 128, 256, 1, 1)
                ->weightUpdateParam(1, 1)
                ->biasUpdateParam(2, 0)
                ->weightFiller(ParamFillerType::Xavier, 0.1)
                ->biasFiller(ParamFillerType::Constant, bias_const)
                ->inputs({"conv3_1"})
                ->outputs({"conv3_2"}))

        ->layer((new typename DropOutLayer<Dtype>::Builder())
                ->id(3005)
                ->probability(0.5)
                ->name("dropout3_2")
                ->inputs({"conv3_2"})
                ->outputs({"conv3_2"}))

        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(3006)
                ->leaky(0.1)
                ->name("relu3_2")
                ->inputs({"conv3_2"})
                ->outputs({"conv3_2"}))

        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(3007)
                ->name("conv3_3")
                ->filterDim(1, 1, 256, 256, 0, 1)
                ->weightUpdateParam(1, 1)
                ->biasUpdateParam(2, 0)
                ->weightFiller(ParamFillerType::Xavier, 0.1)
                ->biasFiller(ParamFillerType::Constant, bias_const)
                ->inputs({"conv3_2"})
                ->outputs({"conv3_3"}))

        ->layer((new typename DropOutLayer<Dtype>::Builder())
                ->id(3008)
                ->probability(0.5)
                ->name("dropout3_3")
                ->inputs({"conv3_3"})
                ->outputs({"conv3_3"}))

        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(3009)
                ->leaky(0.1)
                ->name("relu3_3")
                ->inputs({"conv3_3"})
                ->outputs({"conv3_3"}))

        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(3010)
                ->name("conv3_4")
                ->filterDim(3, 3, 256, 512, 1, 1)
                ->weightUpdateParam(1, 1)
                ->biasUpdateParam(2, 0)
                ->weightFiller(ParamFillerType::Xavier, 0.1)
                ->biasFiller(ParamFillerType::Constant, bias_const)
                ->inputs({"conv3_3"})
                ->outputs({"conv3_4"}))

        ->layer((new typename DropOutLayer<Dtype>::Builder())
                ->id(3011)
                ->probability(0.5)
                ->name("dropout3_4")
                ->inputs({"conv3_4"})
                ->outputs({"conv3_4"}))

        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(3012)
                ->leaky(0.1)
                ->name("relu3_4")
                ->inputs({"conv3_4"})
                ->outputs({"conv3_4"}))

        ->layer((new typename PoolingLayer<Dtype>::Builder())
                ->id(3013)
                ->name("pool3_4")
                ->poolDim(2, 2, 0, 2)
                ->poolingType(Pooling<Dtype>::Max)
                ->inputs({"conv3_4"})
                ->outputs({"pool3_4"}))


        // 4th group
        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(4001)
                ->name("conv4_1")
                ->filterDim(1, 1, 512, 256, 0, 1)
                ->weightUpdateParam(1, 1)
                ->biasUpdateParam(2, 0)
                ->weightFiller(ParamFillerType::Xavier, 0.1)
                ->biasFiller(ParamFillerType::Constant, bias_const)
                ->inputs({"pool3_4"})
                ->outputs({"conv4_1"}))

        ->layer((new typename DropOutLayer<Dtype>::Builder())
                ->id(4002)
                ->probability(0.5)
                ->name("dropout4_1")
                ->inputs({"conv4_1"})
                ->outputs({"conv4_1"}))

        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(4003)
                ->leaky(0.1)
                ->name("relu4_1")
                ->inputs({"conv4_1"})
                ->outputs({"conv4_1"}))

        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(4004)
                ->name("conv4_2")
                ->filterDim(3, 3, 256, 512, 1, 1)
                ->weightUpdateParam(1, 1)
                ->biasUpdateParam(2, 0)
                ->weightFiller(ParamFillerType::Xavier, 0.1)
                ->biasFiller(ParamFillerType::Constant, bias_const)
                ->inputs({"conv4_1"})
                ->outputs({"conv4_2"}))

        ->layer((new typename DropOutLayer<Dtype>::Builder())
                ->id(4005)
                ->probability(0.5)
                ->name("dropout4_2")
                ->inputs({"conv4_2"})
                ->outputs({"conv4_2"}))

        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(4006)
                ->leaky(0.1)
                ->name("relu4_2")
                ->inputs({"conv4_2"})
                ->outputs({"conv4_2"}))

        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(4007)
                ->name("conv4_3")
                ->filterDim(1, 1, 512, 256, 0, 1)
                ->weightUpdateParam(1, 1)
                ->biasUpdateParam(2, 0)
                ->weightFiller(ParamFillerType::Xavier, 0.1)
                ->biasFiller(ParamFillerType::Constant, bias_const)
                ->inputs({"conv4_2"})
                ->outputs({"conv4_3"}))

        ->layer((new typename DropOutLayer<Dtype>::Builder())
                ->id(4008)
                ->probability(0.5)
                ->name("dropout4_3")
                ->inputs({"conv4_3"})
                ->outputs({"conv4_3"}))

        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(4009)
                ->leaky(0.1)
                ->name("relu4_3")
                ->inputs({"conv4_3"})
                ->outputs({"conv4_3"}))

        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(4010)
                ->name("conv4_4")
                ->filterDim(3, 3, 256, 512, 1, 1)
                ->weightUpdateParam(1, 1)
                ->biasUpdateParam(2, 0)
                ->weightFiller(ParamFillerType::Xavier, 0.1)
                ->biasFiller(ParamFillerType::Constant, bias_const)
                ->inputs({"conv4_3"})
                ->outputs({"conv4_4"}))

        ->layer((new typename DropOutLayer<Dtype>::Builder())
                ->id(4011)
                ->probability(0.5)
                ->name("dropout4_4")
                ->inputs({"conv4_4"})
                ->outputs({"conv4_4"}))

        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(4012)
                ->leaky(0.1)
                ->name("relu4_4")
                ->inputs({"conv4_4"})
                ->outputs({"conv4_4"}))

        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(4013)
                ->name("conv4_5")
                ->filterDim(1, 1, 512, 256, 0, 1)
                ->weightUpdateParam(1, 1)
                ->biasUpdateParam(2, 0)
                ->weightFiller(ParamFillerType::Xavier, 0.1)
                ->biasFiller(ParamFillerType::Constant, bias_const)
                ->inputs({"conv4_4"})
                ->outputs({"conv4_5"}))

        ->layer((new typename DropOutLayer<Dtype>::Builder())
                ->id(4014)
                ->probability(0.5)
                ->name("dropout4_5")
                ->inputs({"conv4_5"})
                ->outputs({"conv4_5"}))

        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(4015)
                ->leaky(0.1)
                ->name("relu4_5")
                ->inputs({"conv4_5"})
                ->outputs({"conv4_5"}))

        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(4016)
                ->name("conv4_6")
                ->filterDim(3, 3, 256, 512, 1, 1)
                ->weightUpdateParam(1, 1)
                ->biasUpdateParam(2, 0)
                ->weightFiller(ParamFillerType::Xavier, 0.1)
                ->biasFiller(ParamFillerType::Constant, bias_const)
                ->inputs({"conv4_5"})
                ->outputs({"conv4_6"}))

        ->layer((new typename DropOutLayer<Dtype>::Builder())
                ->id(4017)
                ->probability(0.5)
                ->name("dropout4_6")
                ->inputs({"conv4_6"})
                ->outputs({"conv4_6"}))

        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(4018)
                ->leaky(0.1)
                ->name("relu4_6")
                ->inputs({"conv4_6"})
                ->outputs({"conv4_6"}))


        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(4019)
                ->name("conv4_7")
                ->filterDim(1, 1, 512, 256, 0, 1)
                ->weightUpdateParam(1, 1)
                ->biasUpdateParam(2, 0)
                ->weightFiller(ParamFillerType::Xavier, 0.1)
                ->biasFiller(ParamFillerType::Constant, bias_const)
                ->inputs({"conv4_6"})
                ->outputs({"conv4_7"}))

        ->layer((new typename DropOutLayer<Dtype>::Builder())
                ->id(4020)
                ->probability(0.5)
                ->name("dropout4_7")
                ->inputs({"conv4_7"})
                ->outputs({"conv4_7"}))

        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(4021)
                ->leaky(0.1)
                ->name("relu4_7")
                ->inputs({"conv4_7"})
                ->outputs({"conv4_7"}))

        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(4022)
                ->name("conv4_8")
                ->filterDim(3, 3, 256, 512, 1, 1)
                ->weightUpdateParam(1, 1)
                ->biasUpdateParam(2, 0)
                ->weightFiller(ParamFillerType::Xavier, 0.1)
                ->biasFiller(ParamFillerType::Constant, bias_const)
                ->inputs({"conv4_7"})
                ->outputs({"conv4_8"}))

        ->layer((new typename DropOutLayer<Dtype>::Builder())
                ->id(4023)
                ->probability(0.5)
                ->name("dropout4_8")
                ->inputs({"conv4_8"})
                ->outputs({"conv4_8"}))

        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(4024)
                ->leaky(0.1)
                ->name("relu4_8")
                ->inputs({"conv4_8"})
                ->outputs({"conv4_8"}))

        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(4025)
                ->name("conv4_9")
                ->filterDim(1, 1, 512, 512, 0, 1)
                ->weightUpdateParam(1, 1)
                ->biasUpdateParam(2, 0)
                ->weightFiller(ParamFillerType::Xavier, 0.1)
                ->biasFiller(ParamFillerType::Constant, bias_const)
                ->inputs({"conv4_8"})
                ->outputs({"conv4_9"}))

        ->layer((new typename DropOutLayer<Dtype>::Builder())
                ->id(4026)
                ->probability(0.5)
                ->name("dropout4_9")
                ->inputs({"conv4_9"})
                ->outputs({"conv4_9"}))

        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(4027)
                ->leaky(0.1)
                ->name("relu4_9")
                ->inputs({"conv4_9"})
                ->outputs({"conv4_9"}))

        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(4028)
                ->name("conv4_10")
                ->filterDim(3, 3, 512, 1024, 1, 1)
                ->weightUpdateParam(1, 1)
                ->biasUpdateParam(2, 0)
                ->weightFiller(ParamFillerType::Xavier, 0.1)
                ->biasFiller(ParamFillerType::Constant, bias_const)
                ->inputs({"conv4_9"})
                ->outputs({"conv4_10"}))

        ->layer((new typename DropOutLayer<Dtype>::Builder())
                ->id(4029)
                ->probability(0.5)
                ->name("dropout4_10")
                ->inputs({"conv4_10"})
                ->outputs({"conv4_10"}))

        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(4030)
                ->leaky(0.1)
                ->name("relu4_10")
                ->inputs({"conv4_10"})
                ->outputs({"conv4_10"}))

        ->layer((new typename PoolingLayer<Dtype>::Builder())
                ->id(4031)
                ->name("pool4_10")
                ->poolDim(2, 2, 0, 2)
                ->poolingType(Pooling<Dtype>::Max)
                ->inputs({"conv4_10"})
                ->outputs({"pool4_10"}))

        // 5th group
        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(5001)
                ->name("conv5_1")
                ->filterDim(1, 1, 1024, 512, 0, 1)
                ->weightUpdateParam(1, 1)
                ->biasUpdateParam(2, 0)
                ->weightFiller(ParamFillerType::Xavier, 0.1)
                ->biasFiller(ParamFillerType::Constant, bias_const)
                ->inputs({"pool4_10"})
                ->outputs({"conv5_1"}))

        ->layer((new typename DropOutLayer<Dtype>::Builder())
                ->id(5002)
                ->probability(0.5)
                ->name("dropout5_1")
                ->inputs({"conv5_1"})
                ->outputs({"conv5_1"}))

        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(5003)
                ->leaky(0.1)
                ->name("relu5_1")
                ->inputs({"conv5_1"})
                ->outputs({"conv5_1"}))

        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(5004)
                ->name("conv5_2")
                ->filterDim(3, 3, 512, 1024, 1, 1)
                ->weightUpdateParam(1, 1)
                ->biasUpdateParam(2, 0)
                ->weightFiller(ParamFillerType::Xavier, 0.1)
                ->biasFiller(ParamFillerType::Constant, bias_const)
                ->inputs({"conv5_1"})
                ->outputs({"conv5_2"}))

        ->layer((new typename DropOutLayer<Dtype>::Builder())
                ->id(5005)
                ->probability(0.5)
                ->name("dropout5_2")
                ->inputs({"conv5_2"})
                ->outputs({"conv5_2"}))

        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(5006)
                ->leaky(0.1)
                ->name("relu5_2")
                ->inputs({"conv5_2"})
                ->outputs({"conv5_2"}))

        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(5007)
                ->name("conv5_3")
                ->filterDim(1, 1, 1024, 512, 0, 1)
                ->weightUpdateParam(1, 1)
                ->biasUpdateParam(2, 0)
                ->weightFiller(ParamFillerType::Xavier, 0.1)
                ->biasFiller(ParamFillerType::Constant, bias_const)
                ->inputs({"conv5_2"})
                ->outputs({"conv5_3"}))

        ->layer((new typename DropOutLayer<Dtype>::Builder())
                ->id(5008)
                ->probability(0.5)
                ->name("dropout5_3")
                ->inputs({"conv5_3"})
                ->outputs({"conv5_3"}))

        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(5009)
                ->leaky(0.1)
                ->name("relu5_3")
                ->inputs({"conv5_3"})
                ->outputs({"conv5_3"}))

        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(5010)
                ->name("conv5_4")
                ->filterDim(3, 3, 512, 1024, 1, 1)
                ->weightUpdateParam(1, 1)
                ->biasUpdateParam(2, 0)
                ->weightFiller(ParamFillerType::Xavier, 0.1)
                ->biasFiller(ParamFillerType::Constant, bias_const)
                ->inputs({"conv5_3"})
                ->outputs({"conv5_4"}))

        ->layer((new typename DropOutLayer<Dtype>::Builder())
                ->id(5011)
                ->probability(0.5)
                ->name("dropout5_4")
                ->inputs({"conv5_4"})
                ->outputs({"conv5_4"}))

        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(5012)
                ->leaky(0.1)
                ->name("relu5_4")
                ->inputs({"conv5_4"})
                ->outputs({"conv5_4"}))

        ->layer((new typename PoolingLayer<Dtype>::Builder())
                ->id(6001)
                ->name("pool6_1")
                ->poolDim(2, 2, 0, 2)
                ->poolingType(Pooling<Dtype>::Avg)
                ->inputs({"conv5_4"})
                ->outputs({"pool6_1"}))

        ->layer((new typename FullyConnectedLayer<Dtype>::Builder())
                ->id(6002)
                ->name("fc6_2")
                ->nOut(1000)
                ->weightUpdateParam(1, 1)
                ->biasUpdateParam(2, 0)
                ->weightFiller(ParamFillerType::Gaussian, 0.1)
                ->biasFiller(ParamFillerType::Constant, bias_const)
                ->inputs({"pool6_1"})
                ->outputs({"fc6_2"}))

        ->layer((new typename SoftmaxWithLossLayer<Dtype>::Builder())
                ->id(6003)
                ->name("loss")
                ->inputs({"fc6_2", "label"})
                ->outputs({"loss"}))

        ->build();

    return layersConfig;
}


template<typename Dtype>
LayersConfig<Dtype>* YOLO<Dtype>::createYoloLayersConfig() {
    const float bias_const = 0.2f;

    LayersConfig<Dtype>* layersConfig = (new typename LayersConfig<Dtype>::Builder())
        ->layer((new typename VOCPascalInputLayer<Dtype>::Builder())
                ->id(0)
                ->name("VOCPascalInputLayer")
                ->imageDir(std::string(SPARAM(BASE_DATA_DIR))
                    + std::string("/VOCdevkit/VOCdevkit/"))
                ->source(std::string(SPARAM(BASE_DATA_DIR))
                    + std::string("/VOCdevkit/VOCdevkit/"))
                ->resizeImage(448, 448)
                ->sourceType("ImagePack")
                ->outputs({"data", "label"}))

        // 1st group
        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(1001)
                ->name("conv1_1")
                ->filterDim(7, 7, 3, 64, 1, 2)
                ->weightUpdateParam(1, 1)
                ->biasUpdateParam(2, 0)
                ->weightFiller(ParamFillerType::Xavier, 0.1)
                ->biasFiller(ParamFillerType::Constant, bias_const)
                ->inputs({"data"})
                ->outputs({"conv1_1"}))

        ->layer((new typename DropOutLayer<Dtype>::Builder())
                ->id(1002)
                ->probability(0.5)
                ->name("dropout1_1")
                ->inputs({"conv1_1"})
                ->outputs({"conv1_1"}))

        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(1003)
                ->leaky(0.1)
                ->name("relu1_1")
                ->inputs({"conv1_1"})
                ->outputs({"conv1_1"}))

        ->layer((new typename PoolingLayer<Dtype>::Builder())
                ->id(1004)
                ->name("pool1_1")
                ->poolDim(2, 2, 0, 2)
                ->poolingType(Pooling<Dtype>::Max)
                ->inputs({"conv1_1"})
                ->outputs({"pool1_1"}))


        // 2nd group
        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(2001)
                ->name("conv2_1")
                ->filterDim(3, 3, 64, 192, 1, 1)
                ->weightUpdateParam(1, 1)
                ->biasUpdateParam(2, 0)
                ->weightFiller(ParamFillerType::Xavier, 0.1)
                ->biasFiller(ParamFillerType::Constant, bias_const)
                ->inputs({"pool1_1"})
                ->outputs({"conv2_1"}))

        ->layer((new typename DropOutLayer<Dtype>::Builder())
                ->id(2002)
                ->probability(0.5)
                ->name("dropout2_1")
                ->inputs({"conv2_1"})
                ->outputs({"conv2_1"}))

        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(2003)
                ->leaky(0.1)
                ->name("relu2_1")
                ->inputs({"conv2_1"})
                ->outputs({"conv2_1"}))

        ->layer((new typename PoolingLayer<Dtype>::Builder())
                ->id(2004)
                ->name("pool2_1")
                ->poolDim(2, 2, 0, 2)
                ->poolingType(Pooling<Dtype>::Max)
                ->inputs({"conv2_1"})
                ->outputs({"pool2_1"}))



        // 3rd group
        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(3001)
                ->name("conv3_1")
                ->filterDim(1, 1, 192, 128, 0, 1)
                ->weightUpdateParam(1, 1)
                ->biasUpdateParam(2, 0)
                ->weightFiller(ParamFillerType::Xavier, 0.1)
                ->biasFiller(ParamFillerType::Constant, bias_const)
                ->inputs({"pool2_1"})
                ->outputs({"conv3_1"}))

        ->layer((new typename DropOutLayer<Dtype>::Builder())
                ->id(3002)
                ->probability(0.5)
                ->name("dropout3_1")
                ->inputs({"conv3_1"})
                ->outputs({"conv3_1"}))

        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(3003)
                ->leaky(0.1)
                ->name("relu3_1")
                ->inputs({"conv3_1"})
                ->outputs({"conv3_1"}))

        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(3004)
                ->name("conv3_2")
                ->filterDim(3, 3, 128, 256, 1, 1)
                ->weightUpdateParam(1, 1)
                ->biasUpdateParam(2, 0)
                ->weightFiller(ParamFillerType::Xavier, 0.1)
                ->biasFiller(ParamFillerType::Constant, bias_const)
                ->inputs({"conv3_1"})
                ->outputs({"conv3_2"}))

        ->layer((new typename DropOutLayer<Dtype>::Builder())
                ->id(3005)
                ->probability(0.5)
                ->name("dropout3_2")
                ->inputs({"conv3_2"})
                ->outputs({"conv3_2"}))

        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(3006)
                ->leaky(0.1)
                ->name("relu3_2")
                ->inputs({"conv3_2"})
                ->outputs({"conv3_2"}))

        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(3007)
                ->name("conv3_3")
                ->filterDim(1, 1, 256, 256, 0, 1)
                ->weightUpdateParam(1, 1)
                ->biasUpdateParam(2, 0)
                ->weightFiller(ParamFillerType::Xavier, 0.1)
                ->biasFiller(ParamFillerType::Constant, bias_const)
                ->inputs({"conv3_2"})
                ->outputs({"conv3_3"}))

        ->layer((new typename DropOutLayer<Dtype>::Builder())
                ->id(3008)
                ->probability(0.5)
                ->name("dropout3_3")
                ->inputs({"conv3_3"})
                ->outputs({"conv3_3"}))

        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(3009)
                ->leaky(0.1)
                ->name("relu3_3")
                ->inputs({"conv3_3"})
                ->outputs({"conv3_3"}))

        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(3010)
                ->name("conv3_4")
                ->filterDim(3, 3, 256, 512, 1, 1)
                ->weightUpdateParam(1, 1)
                ->biasUpdateParam(2, 0)
                ->weightFiller(ParamFillerType::Xavier, 0.1)
                ->biasFiller(ParamFillerType::Constant, bias_const)
                ->inputs({"conv3_3"})
                ->outputs({"conv3_4"}))

        ->layer((new typename DropOutLayer<Dtype>::Builder())
                ->id(3011)
                ->probability(0.5)
                ->name("dropout3_4")
                ->inputs({"conv3_4"})
                ->outputs({"conv3_4"}))

        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(3012)
                ->leaky(0.1)
                ->name("relu3_4")
                ->inputs({"conv3_4"})
                ->outputs({"conv3_4"}))

        ->layer((new typename PoolingLayer<Dtype>::Builder())
                ->id(3013)
                ->name("pool3_4")
                ->poolDim(2, 2, 0, 2)
                ->poolingType(Pooling<Dtype>::Max)
                ->inputs({"conv3_4"})
                ->outputs({"pool3_4"}))


        // 4th group
        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(4001)
                ->name("conv4_1")
                ->filterDim(1, 1, 512, 256, 0, 1)
                ->weightUpdateParam(1, 1)
                ->biasUpdateParam(2, 0)
                ->weightFiller(ParamFillerType::Xavier, 0.1)
                ->biasFiller(ParamFillerType::Constant, bias_const)
                ->inputs({"pool3_4"})
                ->outputs({"conv4_1"}))

        ->layer((new typename DropOutLayer<Dtype>::Builder())
                ->id(4002)
                ->probability(0.5)
                ->name("dropout4_1")
                ->inputs({"conv4_1"})
                ->outputs({"conv4_1"}))

        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(4003)
                ->leaky(0.1)
                ->name("relu4_1")
                ->inputs({"conv4_1"})
                ->outputs({"conv4_1"}))

        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(4004)
                ->name("conv4_2")
                ->filterDim(3, 3, 256, 512, 1, 1)
                ->weightUpdateParam(1, 1)
                ->biasUpdateParam(2, 0)
                ->weightFiller(ParamFillerType::Xavier, 0.1)
                ->biasFiller(ParamFillerType::Constant, bias_const)
                ->inputs({"conv4_1"})
                ->outputs({"conv4_2"}))

        ->layer((new typename DropOutLayer<Dtype>::Builder())
                ->id(4005)
                ->probability(0.5)
                ->name("dropout4_2")
                ->inputs({"conv4_2"})
                ->outputs({"conv4_2"}))

        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(4006)
                ->leaky(0.1)
                ->name("relu4_2")
                ->inputs({"conv4_2"})
                ->outputs({"conv4_2"}))

        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(4007)
                ->name("conv4_3")
                ->filterDim(1, 1, 512, 256, 0, 1)
                ->weightUpdateParam(1, 1)
                ->biasUpdateParam(2, 0)
                ->weightFiller(ParamFillerType::Xavier, 0.1)
                ->biasFiller(ParamFillerType::Constant, bias_const)
                ->inputs({"conv4_2"})
                ->outputs({"conv4_3"}))

        ->layer((new typename DropOutLayer<Dtype>::Builder())
                ->id(4008)
                ->probability(0.5)
                ->name("dropout4_3")
                ->inputs({"conv4_3"})
                ->outputs({"conv4_3"}))

        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(4009)
                ->leaky(0.1)
                ->name("relu4_3")
                ->inputs({"conv4_3"})
                ->outputs({"conv4_3"}))

        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(4010)
                ->name("conv4_4")
                ->filterDim(3, 3, 256, 512, 1, 1)
                ->weightUpdateParam(1, 1)
                ->biasUpdateParam(2, 0)
                ->weightFiller(ParamFillerType::Xavier, 0.1)
                ->biasFiller(ParamFillerType::Constant, bias_const)
                ->inputs({"conv4_3"})
                ->outputs({"conv4_4"}))

        ->layer((new typename DropOutLayer<Dtype>::Builder())
                ->id(4011)
                ->probability(0.5)
                ->name("dropout4_4")
                ->inputs({"conv4_4"})
                ->outputs({"conv4_4"}))

        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(4012)
                ->leaky(0.1)
                ->name("relu4_4")
                ->inputs({"conv4_4"})
                ->outputs({"conv4_4"}))

        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(4013)
                ->name("conv4_5")
                ->filterDim(1, 1, 512, 256, 0, 1)
                ->weightUpdateParam(1, 1)
                ->biasUpdateParam(2, 0)
                ->weightFiller(ParamFillerType::Xavier, 0.1)
                ->biasFiller(ParamFillerType::Constant, bias_const)
                ->inputs({"conv4_4"})
                ->outputs({"conv4_5"}))

        ->layer((new typename DropOutLayer<Dtype>::Builder())
                ->id(4014)
                ->probability(0.5)
                ->name("dropout4_5")
                ->inputs({"conv4_5"})
                ->outputs({"conv4_5"}))

        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(4015)
                ->leaky(0.1)
                ->name("relu4_5")
                ->inputs({"conv4_5"})
                ->outputs({"conv4_5"}))

        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(4016)
                ->name("conv4_6")
                ->filterDim(3, 3, 256, 512, 1, 1)
                ->weightUpdateParam(1, 1)
                ->biasUpdateParam(2, 0)
                ->weightFiller(ParamFillerType::Xavier, 0.1)
                ->biasFiller(ParamFillerType::Constant, bias_const)
                ->inputs({"conv4_5"})
                ->outputs({"conv4_6"}))

        ->layer((new typename DropOutLayer<Dtype>::Builder())
                ->id(4017)
                ->probability(0.5)
                ->name("dropout4_6")
                ->inputs({"conv4_6"})
                ->outputs({"conv4_6"}))

        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(4018)
                ->leaky(0.1)
                ->name("relu4_6")
                ->inputs({"conv4_6"})
                ->outputs({"conv4_6"}))


        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(4019)
                ->name("conv4_7")
                ->filterDim(1, 1, 512, 256, 0, 1)
                ->weightUpdateParam(1, 1)
                ->biasUpdateParam(2, 0)
                ->weightFiller(ParamFillerType::Xavier, 0.1)
                ->biasFiller(ParamFillerType::Constant, bias_const)
                ->inputs({"conv4_6"})
                ->outputs({"conv4_7"}))

        ->layer((new typename DropOutLayer<Dtype>::Builder())
                ->id(4020)
                ->probability(0.5)
                ->name("dropout4_7")
                ->inputs({"conv4_7"})
                ->outputs({"conv4_7"}))

        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(4021)
                ->leaky(0.1)
                ->name("relu4_7")
                ->inputs({"conv4_7"})
                ->outputs({"conv4_7"}))

        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(4022)
                ->name("conv4_8")
                ->filterDim(3, 3, 256, 512, 1, 1)
                ->weightUpdateParam(1, 1)
                ->biasUpdateParam(2, 0)
                ->weightFiller(ParamFillerType::Xavier, 0.1)
                ->biasFiller(ParamFillerType::Constant, bias_const)
                ->inputs({"conv4_7"})
                ->outputs({"conv4_8"}))

        ->layer((new typename DropOutLayer<Dtype>::Builder())
                ->id(4023)
                ->probability(0.5)
                ->name("dropout4_8")
                ->inputs({"conv4_8"})
                ->outputs({"conv4_8"}))

        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(4024)
                ->leaky(0.1)
                ->name("relu4_8")
                ->inputs({"conv4_8"})
                ->outputs({"conv4_8"}))

        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(4025)
                ->name("conv4_9")
                ->filterDim(1, 1, 512, 512, 0, 1)
                ->weightUpdateParam(1, 1)
                ->biasUpdateParam(2, 0)
                ->weightFiller(ParamFillerType::Xavier, 0.1)
                ->biasFiller(ParamFillerType::Constant, bias_const)
                ->inputs({"conv4_8"})
                ->outputs({"conv4_9"}))

        ->layer((new typename DropOutLayer<Dtype>::Builder())
                ->id(4026)
                ->probability(0.5)
                ->name("dropout4_9")
                ->inputs({"conv4_9"})
                ->outputs({"conv4_9"}))

        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(4027)
                ->leaky(0.1)
                ->name("relu4_9")
                ->inputs({"conv4_9"})
                ->outputs({"conv4_9"}))

        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(4028)
                ->name("conv4_10")
                ->filterDim(3, 3, 512, 1024, 1, 1)
                ->weightUpdateParam(1, 1)
                ->biasUpdateParam(2, 0)
                ->weightFiller(ParamFillerType::Xavier, 0.1)
                ->biasFiller(ParamFillerType::Constant, bias_const)
                ->inputs({"conv4_9"})
                ->outputs({"conv4_10"}))

        ->layer((new typename DropOutLayer<Dtype>::Builder())
                ->id(4029)
                ->probability(0.5)
                ->name("dropout4_10")
                ->inputs({"conv4_10"})
                ->outputs({"conv4_10"}))

        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(4030)
                ->leaky(0.1)
                ->name("relu4_10")
                ->inputs({"conv4_10"})
                ->outputs({"conv4_10"}))

        ->layer((new typename PoolingLayer<Dtype>::Builder())
                ->id(4031)
                ->name("pool4_10")
                ->poolDim(2, 2, 0, 2)
                ->poolingType(Pooling<Dtype>::Max)
                ->inputs({"conv4_10"})
                ->outputs({"pool4_10"}))

        // 5th group
        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(5001)
                ->name("conv5_1")
                ->filterDim(1, 1, 1024, 512, 0, 1)
                ->weightUpdateParam(1, 1)
                ->biasUpdateParam(2, 0)
                ->weightFiller(ParamFillerType::Xavier, 0.1)
                ->biasFiller(ParamFillerType::Constant, bias_const)
                ->inputs({"pool4_10"})
                ->outputs({"conv5_1"}))

        ->layer((new typename DropOutLayer<Dtype>::Builder())
                ->id(5002)
                ->probability(0.5)
                ->name("dropout5_1")
                ->inputs({"conv5_1"})
                ->outputs({"conv5_1"}))

        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(5003)
                ->leaky(0.1)
                ->name("relu5_1")
                ->inputs({"conv5_1"})
                ->outputs({"conv5_1"}))

        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(5004)
                ->name("conv5_2")
                ->filterDim(3, 3, 512, 1024, 1, 1)
                ->weightUpdateParam(1, 1)
                ->biasUpdateParam(2, 0)
                ->weightFiller(ParamFillerType::Xavier, 0.1)
                ->biasFiller(ParamFillerType::Constant, bias_const)
                ->inputs({"conv5_1"})
                ->outputs({"conv5_2"}))

        ->layer((new typename DropOutLayer<Dtype>::Builder())
                ->id(5005)
                ->probability(0.5)
                ->name("dropout5_2")
                ->inputs({"conv5_2"})
                ->outputs({"conv5_2"}))

        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(5006)
                ->leaky(0.1)
                ->name("relu5_2")
                ->inputs({"conv5_2"})
                ->outputs({"conv5_2"}))

        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(5007)
                ->name("conv5_3")
                ->filterDim(1, 1, 1024, 512, 0, 1)
                ->weightUpdateParam(1, 1)
                ->biasUpdateParam(2, 0)
                ->weightFiller(ParamFillerType::Xavier, 0.1)
                ->biasFiller(ParamFillerType::Constant, bias_const)
                ->inputs({"conv5_2"})
                ->outputs({"conv5_3"}))

        ->layer((new typename DropOutLayer<Dtype>::Builder())
                ->id(5008)
                ->probability(0.5)
                ->name("dropout5_3")
                ->inputs({"conv5_3"})
                ->outputs({"conv5_3"}))

        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(5009)
                ->leaky(0.1)
                ->name("relu5_3")
                ->inputs({"conv5_3"})
                ->outputs({"conv5_3"}))

        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(5010)
                ->name("conv5_4")
                ->filterDim(3, 3, 512, 1024, 1, 1)
                ->weightUpdateParam(1, 1)
                ->biasUpdateParam(2, 0)
                ->weightFiller(ParamFillerType::Xavier, 0.1)
                ->biasFiller(ParamFillerType::Constant, bias_const)
                ->inputs({"conv5_3"})
                ->outputs({"conv5_4"}))

        ->layer((new typename DropOutLayer<Dtype>::Builder())
                ->id(5011)
                ->probability(0.5)
                ->name("dropout5_4")
                ->inputs({"conv5_4"})
                ->outputs({"conv5_4"}))

        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(5012)
                ->leaky(0.1)
                ->name("relu5_4")
                ->inputs({"conv5_4"})
                ->outputs({"conv5_4"}))

        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(5013)
                ->name("conv5_5")
                ->filterDim(3, 3, 1024, 1024, 1, 1)
                ->weightUpdateParam(1, 1)
                ->biasUpdateParam(2, 0)
                ->weightFiller(ParamFillerType::Xavier, 0.1)
                ->biasFiller(ParamFillerType::Constant, bias_const)
                ->inputs({"conv5_4"})
                ->outputs({"conv5_5"}))

        ->layer((new typename DropOutLayer<Dtype>::Builder())
                ->id(5014)
                ->probability(0.5)
                ->name("dropout5_5")
                ->inputs({"conv5_5"})
                ->outputs({"conv5_5"}))

        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(5015)
                ->leaky(0.1)
                ->name("relu5_5")
                ->inputs({"conv5_5"})
                ->outputs({"conv5_5"}))

        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(5016)
                ->name("conv5_6")
                ->filterDim(3, 3, 1024, 1024, 1, 2)
                ->weightUpdateParam(1, 1)
                ->biasUpdateParam(2, 0)
                ->weightFiller(ParamFillerType::Xavier, 0.1)
                ->biasFiller(ParamFillerType::Constant, bias_const)
                ->inputs({"conv5_5"})
                ->outputs({"conv5_6"}))

        ->layer((new typename DropOutLayer<Dtype>::Builder())
                ->id(5017)
                ->probability(0.5)
                ->name("dropout5_6")
                ->inputs({"conv5_6"})
                ->outputs({"conv5_6"}))

        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(5018)
                ->leaky(0.1)
                ->name("relu5_6")
                ->inputs({"conv5_6"})
                ->outputs({"conv5_6"}))

        // 6th group
        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(6001)
                ->name("conv6_1")
                ->filterDim(3, 3, 1024, 1024, 1, 1)
                ->weightUpdateParam(1, 1)
                ->biasUpdateParam(2, 0)
                ->weightFiller(ParamFillerType::Xavier, 0.1)
                ->biasFiller(ParamFillerType::Constant, bias_const)
                ->inputs({"conv5_6"})
                ->outputs({"conv6_1"}))

        ->layer((new typename DropOutLayer<Dtype>::Builder())
                ->id(6002)
                ->probability(0.5)
                ->name("dropout6_1")
                ->inputs({"conv6_1"})
                ->outputs({"conv6_1"}))

        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(6003)
                ->leaky(0.1)
                ->name("relu6_1")
                ->inputs({"conv6_1"})
                ->outputs({"conv6_1"}))

        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(6004)
                ->name("conv6_2")
                ->filterDim(3, 3, 1024, 1024, 1, 1)
                ->weightUpdateParam(1, 1)
                ->biasUpdateParam(2, 0)
                ->weightFiller(ParamFillerType::Xavier, 0.1)
                ->biasFiller(ParamFillerType::Constant, bias_const)
                ->inputs({"conv6_1"})
                ->outputs({"conv6_2"}))

        ->layer((new typename DropOutLayer<Dtype>::Builder())
                ->id(6005)
                ->probability(0.5)
                ->name("dropout6_2")
                ->inputs({"conv6_2"})
                ->outputs({"conv6_2"}))

        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(6006)
                ->leaky(0.1)
                ->name("relu6_2")
                ->inputs({"conv6_2"})
                ->outputs({"conv6_2"}))


        ->layer((new typename FullyConnectedLayer<Dtype>::Builder())
                ->id(7001)
                ->name("fc7_1")
                ->nOut(4096)
                ->weightUpdateParam(1, 1)
                ->biasUpdateParam(2, 0)
                ->weightFiller(ParamFillerType::Gaussian, 0.1)
                ->biasFiller(ParamFillerType::Constant, bias_const)
                ->inputs({"conv6_2"})
                ->outputs({"fc7_1"}))

        ->layer((new typename DropOutLayer<Dtype>::Builder())
                ->id(7002)
                ->probability(0.5)
                ->name("dropout7_1")
                ->inputs({"fc7_1"})
                ->outputs({"fc7_1"}))

        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(7003)
                ->leaky(0.1)
                ->name("relu7_1")
                ->inputs({"fc7_1"})
                ->outputs({"fc7_1"}))

        ->layer((new typename FullyConnectedLayer<Dtype>::Builder())
                ->id(8001)
                ->name("fc8_1")
                ->nOut(1470)        // 7 * 7 * 30
                ->weightUpdateParam(1, 1)
                ->biasUpdateParam(2, 0)
                ->weightFiller(ParamFillerType::Gaussian, 0.1)
                ->biasFiller(ParamFillerType::Constant, bias_const)
                ->inputs({"fc7_1"})
                ->outputs({"fc8_1"}))

        ->layer((new typename DropOutLayer<Dtype>::Builder())
                ->id(8002)
                ->probability(0.5)
                ->name("dropout8_1")
                ->inputs({"fc8_1"})
                ->outputs({"fc8_1"}))

        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(8003)
                ->name("relu8_1")
                ->inputs({"fc8_1"})
                ->outputs({"fc8_1"}))

        ->layer((new typename YOLOOutputLayer<Dtype>::Builder())
                ->id(9001)
                ->name("loss")
                ->inputs({"fc8_1", "label"})
                ->outputs({"loss"}))

        ->build();

    return layersConfig;
}

template<typename Dtype>
void YOLO<Dtype>::runPretrain() {
    // loss layer of Discriminator GAN 
	const vector<string> lossList = { "loss" };
    // loss layer of Generatoer-Discriminator 0 GAN

	const NetworkPhase phase = NetworkPhase::TrainPhase;
	const uint32_t batchSize = 16;
	const uint32_t testInterval = 1;		// 10000(목표 샘플수) / batchSize
	const uint32_t saveInterval = 100000;		// 1000000 / batchSize
	const float baseLearningRate = 0.001f;  // 0.001 ~ 0.01 (paper recommendation)

	const uint32_t stepSize = 100000;
	const float weightDecay = 0.0005f;
	const float momentum = 0.9f;
	const float clipGradientsLevel = 0.0f;
	const LRPolicy lrPolicy = LRPolicy::Fixed;

    //const Optimizer opt = Optimizer::Adam;
    const Optimizer opt = Optimizer::Momentum;

	STDOUT_BLOCK(cout << "batchSize: " << batchSize << endl;);
	STDOUT_BLOCK(cout << "testInterval: " << testInterval << endl;);
	STDOUT_BLOCK(cout << "saveInterval: " << saveInterval << endl;);
	STDOUT_BLOCK(cout << "baseLearningRate: " << baseLearningRate << endl;);
	STDOUT_BLOCK(cout << "weightDecay: " << weightDecay << endl;);
	STDOUT_BLOCK(cout << "momentum: " << momentum << endl;);
	STDOUT_BLOCK(cout << "clipGradientsLevel: " << clipGradientsLevel << endl;);

	NetworkConfig<Dtype>* networkConfig =
			(new typename NetworkConfig<Dtype>::Builder())
			->batchSize(batchSize)
			->baseLearningRate(baseLearningRate)
			->weightDecay(weightDecay)
			->momentum(momentum)
			->testInterval(testInterval)
			->saveInterval(saveInterval)
			->stepSize(stepSize)
			->clipGradientsLevel(clipGradientsLevel)
			->lrPolicy(lrPolicy)
			->networkPhase(phase)
			->savePathPrefix(SPARAM(NETWORK_SAVE_DIR))
			->networkListeners({
				new NetworkMonitor("loss", NetworkMonitor::PLOT_ONLY),
				})
			->lossLayers(lossList)
            ->optimizer(opt)
			->build();

	Util::printVramInfo();

 	Network<Dtype>* network = new Network<Dtype>(networkConfig);

    // (1) layer config를 만든다. 이 과정중에 layer들의 초기화가 진행된다.
	LayersConfig<Dtype>* layersConfig = createYoloPreLayersConfig();
 	network->setLayersConfig(layersConfig);

	// (2) network config 정보를 layer들에게 전달한다.
	for (uint32_t i = 0; i < layersConfig->_layers.size(); i++)
		layersConfig->_layers[i]->setNetworkConfig(networkConfig);

    network->sgd(1);
    networkConfig->save();
    DebugUtil<Dtype>::printNetworkEdges(stderr, "save network", layersConfig, 0);
}

template<typename Dtype>
void YOLO<Dtype>::run() {
    // loss layer of Discriminator GAN 
	const vector<string> lossList = { "loss" };
    // loss layer of Generatoer-Discriminator 0 GAN

	const NetworkPhase phase = NetworkPhase::TrainPhase;
    vector<WeightsArg> weightsArgs(1);
    weightsArgs[0].weightsPath =
        string(SPARAM(NETWORK_SAVE_DIR)) + "/network80072.param";

	const uint32_t batchSize = 8;
	const uint32_t testInterval = 1;		// 10000(목표 샘플수) / batchSize
	const uint32_t saveInterval = 100000;		// 1000000 / batchSize
	const float baseLearningRate = 0.001f;  // 0.001 ~ 0.01 (paper recommendation)

	const uint32_t stepSize = 100000;
	const float weightDecay = 0.0005f;
	const float momentum = 0.9f;
	const float clipGradientsLevel = 0.0f;
	const LRPolicy lrPolicy = LRPolicy::Fixed;

    //const Optimizer opt = Optimizer::Adam;
    const Optimizer opt = Optimizer::Momentum;

	STDOUT_BLOCK(cout << "batchSize: " << batchSize << endl;);
	STDOUT_BLOCK(cout << "testInterval: " << testInterval << endl;);
	STDOUT_BLOCK(cout << "saveInterval: " << saveInterval << endl;);
	STDOUT_BLOCK(cout << "baseLearningRate: " << baseLearningRate << endl;);
	STDOUT_BLOCK(cout << "weightDecay: " << weightDecay << endl;);
	STDOUT_BLOCK(cout << "momentum: " << momentum << endl;);
	STDOUT_BLOCK(cout << "clipGradientsLevel: " << clipGradientsLevel << endl;);

	NetworkConfig<Dtype>* networkConfig =
			(new typename NetworkConfig<Dtype>::Builder())
			->batchSize(batchSize)
			->baseLearningRate(baseLearningRate)
			->weightDecay(weightDecay)
			->momentum(momentum)
			->testInterval(testInterval)
			->saveInterval(saveInterval)
			->stepSize(stepSize)
			->clipGradientsLevel(clipGradientsLevel)
			->lrPolicy(lrPolicy)
			->networkPhase(phase)
			->savePathPrefix(SPARAM(NETWORK_SAVE_DIR))
            ->weightsArgs(weightsArgs)
			->networkListeners({
				new NetworkMonitor("loss", NetworkMonitor::PLOT_ONLY),
				})
			->lossLayers(lossList)
            ->optimizer(opt)
			->build();

	Util::printVramInfo();

 	Network<Dtype>* network = new Network<Dtype>(networkConfig);

    // (1) layer config를 만든다. 이 과정중에 layer들의 초기화가 진행된다.
	LayersConfig<Dtype>* layersConfig = createYoloLayersConfig();

	// (2) network config 정보를 layer들에게 전달한다.
	for (uint32_t i = 0; i < layersConfig->_layers.size(); i++)
		layersConfig->_layers[i]->setNetworkConfig(networkConfig);

 	network->setLayersConfig(layersConfig);
    //network->loadPretrainedWeights();

    network->sgd(1);
    //DebugUtil<Dtype>::printNetworkEdges(stderr, "load network", layersConfig, 0);
}

template class YOLO<float>;
