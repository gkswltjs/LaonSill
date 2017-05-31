/**
 * @file GAN.cpp
 * @date 2017-04-20
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <vector>

#include "common.h"
#include "Debug.h"
#include "GAN.h"
#include "StdOutLog.h"
#include "Network.h"
#include "NetworkMonitor.h"
#include "ImageUtil.h"

using namespace std;

template<typename Dtype>
void GAN<Dtype>::setLayerTrain(LayersConfig<Dtype>* lc, bool train) {
#if 0
    int layerCount = lc->_layers.size();

    for (int i = 0; i < layerCount; i++) {
        BatchNormLayer<Dtype>* bnLayer = dynamic_cast<BatchNormLayer<Dtype>*>(lc->_layers[i]);

        if (bnLayer == NULL)
            continue;

        bnLayer->setTrain(train);
    }
#endif
}

template <typename Dtype>
LayersConfig<Dtype>* GAN<Dtype>::createDOfGANLayersConfig() {
#if 0
	LayersConfig<Dtype>* layersConfig =
	    (new typename LayersConfig<Dtype>::Builder())

        ->layer((new typename CelebAInputLayer<Dtype>::Builder())
                ->id(1)
                ->name("CelebAInputLayer")
                ->imageDir(std::string(SPARAM(BASE_DATA_DIR))
                    + std::string("/celebA"))
                ->cropImage(108)
                ->resizeImage(64,64)
                ->outputs({"data"}))

        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(2)
                ->name("ConvLayer1")
                ->filterDim(4, 4, 3, 64, 1, 2)
                ->weightUpdateParam(1, 0)
                ->biasUpdateParam(1, 0)
                ->weightFiller(ParamFillerType::Gaussian, 0.02)
                ->biasFiller(ParamFillerType::Constant, 0.0)
                ->inputs({"data"})
                ->outputs({"conv1"})
                ->receive(10015))
        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(4)
                ->leaky(0.2)
                ->name("LeakyRelu1")
                ->inputs({"conv1"})
                ->outputs({"lrelu1"}))
        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(5)
                ->name("ConvLayer2")
                ->filterDim(4, 4, 64, 128, 1, 2)
                ->weightUpdateParam(1, 0)
                ->biasUpdateParam(1, 0)
                ->weightFiller(ParamFillerType::Gaussian, 0.02)
                ->biasFiller(ParamFillerType::Constant, 0.0)
                ->inputs({"lrelu1"})
                ->outputs({"conv2"})
                ->receive(10018))
        ->layer((new typename BatchNormLayer<Dtype>::Builder())
                ->id(6)
                ->name("BNLayer/conv2")
                ->inputs({"conv2"})
                ->outputs({"BN/conv2"})
                ->receive(10019))
        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(7)
                ->leaky(0.2)
                ->name("LeakyRelu2")
                ->inputs({"BN/conv2"})
                ->outputs({"lrelu2"}))
        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(8)
                ->name("convLayer3")
                ->filterDim(4, 4, 128, 256, 1, 2)
                ->weightUpdateParam(1, 0)
                ->biasUpdateParam(1, 0)
                ->weightFiller(ParamFillerType::Gaussian, 0.02)
                ->biasFiller(ParamFillerType::Constant, 0.0)
                ->inputs({"lrelu2"})
                ->outputs({"conv3"})
                ->receive(10021))
        ->layer((new typename BatchNormLayer<Dtype>::Builder())
                ->id(9)
                ->name("BNLayer/conv3")
                ->inputs({"conv3"})
                ->outputs({"BN/conv3"})
                ->receive(10022))
        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(10)
                ->leaky(0.2)
                ->name("LeakyRelu3")
                ->inputs({"BN/conv3"})
                ->outputs({"lrelu3"}))
        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(11)
                ->name("convLayer4")
                ->filterDim(4, 4, 256, 512, 1, 2)
                ->weightUpdateParam(1, 0)
                ->biasUpdateParam(1, 0)
                ->weightFiller(ParamFillerType::Gaussian, 0.02)
                ->biasFiller(ParamFillerType::Constant, 0.0)
                ->inputs({"lrelu3"})
                ->outputs({"conv4"})
                ->receive(10024))
        ->layer((new typename BatchNormLayer<Dtype>::Builder())
                ->id(12)
                ->name("BNLayer/conv4")
                ->inputs({"conv4"})
                ->outputs({"BN/conv4"})
                ->receive(10025))
        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(13)
                ->leaky(0.2)
                ->name("LeakyRelu4")
                ->inputs({"BN/conv4"})
                ->outputs({"lrelu4"}))
        ->layer((new typename FullyConnectedLayer<Dtype>::Builder())
                ->id(14)
                ->name("fc1")
                ->nOut(1)
                ->weightUpdateParam(1, 0)
                ->biasUpdateParam(1, 0)
                ->weightFiller(ParamFillerType::Gaussian, 0.02)
                ->biasFiller(ParamFillerType::Constant, 0.0)
                ->inputs({"lrelu4"})
                ->outputs({"fc1"})
                ->receive(10027))
        ->layer((new typename CrossEntropyWithLossLayer<Dtype>::Builder())
                ->id(15)
                ->targetValue(1.0)
                ->withSigmoid(true)
                ->name("celossDGAN")
                ->inputs({"fc1"})
                ->outputs({"prob"}))
        ->build();

	return layersConfig;
#else
    return NULL;
#endif
}

template<typename Dtype>
LayersConfig<Dtype>* GAN<Dtype>::createGD0OfGANLayersConfig() {
#if 0
	LayersConfig<Dtype>* layersConfig =
	    (new typename LayersConfig<Dtype>::Builder())
        ->layer((new typename NoiseInputLayer<Dtype>::Builder())
                ->id(9999)
                ->name("NoiseInputLayer")
                ->noise(100, -1.0, 1.0)
                ->outputs({"noise"})
                )
        ->layer((new typename FullyConnectedLayer<Dtype>::Builder())
                ->id(10000)
                ->name("fc0")
                ->nOut(4 * 4 * 512)
                ->weightUpdateParam(1, 0)
                ->biasUpdateParam(1, 0)
                ->weightFiller(ParamFillerType::Gaussian, 0.02)
                ->biasFiller(ParamFillerType::Constant, 0.0)
                ->inputs({"noise"})
                ->outputs({"fc0"}))
        ->layer((new typename ReshapeLayer<Dtype>::Builder())
                ->id(10001)
                ->name("reshape")
                ->shape({-1, 512, 4, 4})
                ->inputs({"fc0"})
                ->outputs({"reshape0"}))
        ->layer((new typename BatchNormLayer<Dtype>::Builder())
                ->id(10002)
                ->name("BNLayer/noiseInput")
                ->inputs({"reshape0"})
                ->outputs({"BN/noiseInput"}))
        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(10003)
                ->name("ReluForNoise")
                ->inputs({"BN/noiseInput"})
                ->outputs({"relu/noiseInput"}))
        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(10004)
                ->name("DeconvLayer1")
                ->filterDim(4, 4, 512, 256, 1, 2)
                ->weightUpdateParam(1, 0)
                ->biasUpdateParam(1, 0)
                ->weightFiller(ParamFillerType::Gaussian, 0.02)
                ->biasFiller(ParamFillerType::Constant, 0.0)
                ->inputs({"relu/noiseInput"})
                ->outputs({"deconv1"})
                ->deconv(true))
        ->layer((new typename BatchNormLayer<Dtype>::Builder())
                ->id(10005)
                ->name("BNLayer/deconv1")
                ->inputs({"deconv1"})
                ->outputs({"BN/deconv1"}))
        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(10006)
                ->name("Relu1")
                ->inputs({"BN/deconv1"})
                ->outputs({"relu1"}))
        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(10007)
                ->name("DeconvLayer2")
                ->filterDim(4, 4, 256, 128, 1, 2)
                ->weightUpdateParam(1, 0)
                ->biasUpdateParam(1, 0)
                ->weightFiller(ParamFillerType::Gaussian, 0.02)
                ->biasFiller(ParamFillerType::Constant, 0.0)
                ->inputs({"relu1"})
                ->outputs({"deconv2"})
                ->deconv(true))
        ->layer((new typename BatchNormLayer<Dtype>::Builder())
                ->id(10008)
                ->name("BNLayer/deconv2")
                ->inputs({"deconv2"})
                ->outputs({"BN/deconv2"}))
        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(10009)
                ->name("Relu2")
                ->inputs({"BN/deconv2"})
                ->outputs({"relu2"}))
        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(10010)
                ->name("DeconvLayer3")
                ->filterDim(4, 4, 128, 64, 1, 2)
                ->weightUpdateParam(1, 0)
                ->biasUpdateParam(1, 0)
                ->weightFiller(ParamFillerType::Gaussian, 0.02)
                ->biasFiller(ParamFillerType::Constant, 0.0)
                ->inputs({"relu2"})
                ->outputs({"deconv3"})
                ->deconv(true))
        ->layer((new typename BatchNormLayer<Dtype>::Builder())
                ->id(10011)
                ->name("BNLayer/deconv3")
                ->inputs({"deconv3"})
                ->outputs({"BN/deconv3"}))
        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(10012)
                ->name("Relu3")
                ->inputs({"BN/deconv3"})
                ->outputs({"relu3"}))
        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(10013)
                ->name("DeconvLayer4")
                ->filterDim(4, 4, 64, 3, 1, 2)
                ->weightUpdateParam(1, 0)
                ->biasUpdateParam(1, 0)
                ->weightFiller(ParamFillerType::Gaussian, 0.02)
                ->biasFiller(ParamFillerType::Constant, 0.0)
                ->inputs({"relu3"})
                ->outputs({"deconv4"})
                ->deconv(true))

        ->layer((new typename HyperTangentLayer<Dtype>::Builder())
                ->id(10014)
                ->name("hypertangent")
                ->inputs({"deconv4"})
                ->outputs({"hypertangent"}))
        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(10015)
                ->name("ConvLayer1")
                ->filterDim(4, 4, 3, 64, 1, 2)
                ->weightUpdateParam(1, 0)
                ->biasUpdateParam(1, 0)
                ->weightFiller(ParamFillerType::Gaussian, 0.02)
                ->biasFiller(ParamFillerType::Constant, 0.0)
                ->inputs({"hypertangent"})
                ->outputs({"conv1"})
                ->donate())
        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(10017)
                ->leaky(0.2)
                ->name("LeakyRelu1")
                ->inputs({"conv1"})
                ->outputs({"lrelu1"}))
        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(10018)
                ->name("ConvLayer2")
                ->filterDim(4, 4, 64, 128, 1, 2)
                ->weightUpdateParam(1, 0)
                ->biasUpdateParam(1, 0)
                ->weightFiller(ParamFillerType::Gaussian, 0.02)
                ->biasFiller(ParamFillerType::Constant, 0.0)
                ->inputs({"lrelu1"})
                ->outputs({"conv2"})
                ->donate())
        ->layer((new typename BatchNormLayer<Dtype>::Builder())
                ->id(10019)
                ->name("BNLayer/conv2")
                ->inputs({"conv2"})
                ->outputs({"BN/conv2"})
                ->donate())
        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(10020)
                ->leaky(0.2)
                ->name("LeakyRelu2")
                ->inputs({"BN/conv2"})
                ->outputs({"lrelu2"}))
        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(10021)
                ->name("convLayer3")
                ->filterDim(4, 4, 128, 256, 1, 2)
                ->weightUpdateParam(1, 0)
                ->biasUpdateParam(1, 0)
                ->weightFiller(ParamFillerType::Gaussian, 0.02)
                ->biasFiller(ParamFillerType::Constant, 0.0)
                ->inputs({"lrelu2"})
                ->outputs({"conv3"})
                ->donate())
        ->layer((new typename BatchNormLayer<Dtype>::Builder())
                ->id(10022)
                ->name("BNLayer/conv3")
                ->inputs({"conv3"})
                ->outputs({"BN/conv3"})
                ->donate())
        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(10023)
                ->leaky(0.2)
                ->name("LeakyRelu3")
                ->inputs({"BN/conv3"})
                ->outputs({"lrelu3"}))
        ->layer((new typename ConvLayer<Dtype>::Builder())
                ->id(10024)
                ->name("convLayer4")
                ->filterDim(4, 4, 256, 512, 1, 2)
                ->weightUpdateParam(1, 0)
                ->biasUpdateParam(1, 0)
                ->weightFiller(ParamFillerType::Gaussian, 0.02)
                ->biasFiller(ParamFillerType::Constant, 0.0)
                ->inputs({"lrelu3"})
                ->outputs({"conv4"})
                ->donate())
        ->layer((new typename BatchNormLayer<Dtype>::Builder())
                ->id(10025)
                ->name("BNLayer/conv4")
                ->inputs({"conv4"})
                ->outputs({"BN/conv4"})
                ->donate())
        ->layer((new typename ReluLayer<Dtype>::Builder())
                ->id(10026)
                ->leaky(0.2)
                ->name("LeakyRelu4")
                ->inputs({"BN/conv4"})
                ->outputs({"lrelu4"}))
        ->layer((new typename FullyConnectedLayer<Dtype>::Builder())
                ->id(10027)
                ->name("fc1")
                ->nOut(1)
                ->weightUpdateParam(1, 0)
                ->biasUpdateParam(1, 0)
                ->weightFiller(ParamFillerType::Gaussian, 0.02)
                ->biasFiller(ParamFillerType::Constant, 0.0)
                ->inputs({"lrelu4"})
                ->outputs({"fc1"})
                ->donate())
        ->layer((new typename CrossEntropyWithLossLayer<Dtype>::Builder())
                ->id(10028)
                ->targetValue(0.0)
                ->withSigmoid(true)
                ->name("celossGD0GAN")
                ->inputs({"fc1"})
                ->outputs({"prob"}))
        ->build();

	return layersConfig;
#else
    return NULL;
#endif
}

template<typename Dtype>
void GAN<Dtype>::run() {
#if 0
    // loss layer of Discriminator GAN 
	const vector<string> llDGAN = { "celossDGAN" };
    // loss layer of Generatoer-Discriminator 0 GAN
	const vector<string> llGD0GAN = { "celossGD0GAN" };

	const NetworkPhase phase = NetworkPhase::TrainPhase;
	const uint32_t batchSize = 64;
	const uint32_t testInterval = 1;		// 10000(목표 샘플수) / batchSize
	const uint32_t saveInterval = 100000;		// 1000000 / batchSize
	const float baseLearningRate = 0.0002f;

	const uint32_t stepSize = 100000;
	const float weightDecay = 0.0001f;
	const float momentum = 0.9f;
	const float clipGradientsLevel = 0.0f;
	const float gamma = 0.1;
	const LRPolicy lrPolicy = LRPolicy::Fixed;

    const Optimizer opt = Optimizer::Adam;
    //const Optimizer opt = Optimizer::Momentum;

	STDOUT_BLOCK(cout << "batchSize: " << batchSize << endl;);
	STDOUT_BLOCK(cout << "testInterval: " << testInterval << endl;);
	STDOUT_BLOCK(cout << "saveInterval: " << saveInterval << endl;);
	STDOUT_BLOCK(cout << "baseLearningRate: " << baseLearningRate << endl;);
	STDOUT_BLOCK(cout << "weightDecay: " << weightDecay << endl;);
	STDOUT_BLOCK(cout << "momentum: " << momentum << endl;);
	STDOUT_BLOCK(cout << "clipGradientsLevel: " << clipGradientsLevel << endl;);

	NetworkConfig<Dtype>* ncDGAN =
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
			->gamma(gamma)
			->savePathPrefix(SPARAM(NETWORK_SAVE_DIR))
			->networkListeners({
				new NetworkMonitor("celossDGAN", NetworkMonitor::PLOT_ONLY),
				})
			->lossLayers(llDGAN)
            ->optimizer(opt)
            ->beta(0.5, 0.999)
			->build();

	NetworkConfig<Dtype>* ncGD0GAN =
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
			->gamma(gamma)
			->savePathPrefix(SPARAM(NETWORK_SAVE_DIR))
			->networkListeners({
				new NetworkMonitor("celossGD0GAN", NetworkMonitor::PLOT_ONLY),
				})
			->lossLayers(llGD0GAN)
            ->optimizer(opt)
            ->beta(0.5, 0.999)
			->build();

	Util::printVramInfo();

 	Network<Dtype>* networkDGAN = new Network<Dtype>(ncDGAN);
 	Network<Dtype>* networkGD0GAN = new Network<Dtype>(ncGD0GAN);

    // (1) layer config를 만든다. 이 과정중에 layer들의 초기화가 진행된다.
	LayersConfig<Dtype>* lcGD0GAN = createGD0OfGANLayersConfig();
	LayersConfig<Dtype>* lcDGAN = createDOfGANLayersConfig();
 	networkGD0GAN->setLayersConfig(lcGD0GAN);
 	networkDGAN->setLayersConfig(lcDGAN);

	// (2) network config 정보를 layer들에게 전달한다.
	for (uint32_t i = 0; i < lcDGAN->_layers.size(); i++)
		lcDGAN->_layers[i]->setNetworkConfig(ncDGAN);

	for (uint32_t i = 0; i < lcGD0GAN->_layers.size(); i++)
		lcGD0GAN->_layers[i]->setNetworkConfig(ncGD0GAN);

    for (int i = 0; i < 100000; i++) {  // epoch
        cout << "epoch=" << i << endl;

        InputLayer<Dtype>* inputLayer = lcDGAN->_inputLayer;
        inputLayer->reshape();  // to fill training data
        const uint32_t trainDataSize = inputLayer->getNumTrainData();
        const uint32_t numBatches = trainDataSize / ncDGAN->_batchSize - 1;

        for (int j = 0; j < numBatches; j++) {
            // feedforward
            for (int loop = 0; loop < 1; loop++) {
                networkGD0GAN->_feedforward(0);
                networkGD0GAN->_backpropagation(0);
                for (uint32_t k = 9; k < lcGD0GAN->_learnableLayers.size(); k++) {
                    lcGD0GAN->_learnableLayers[k]->update();
                }

                networkDGAN->_feedforward(j);
                networkDGAN->_backpropagation(j);
                for (uint32_t k = 0; k < lcDGAN->_learnableLayers.size(); k++) {
                    lcDGAN->_learnableLayers[k]->update();
                }

                // calculate D loss
                {
                    CrossEntropyWithLossLayer<Dtype>* lossDRealLayer =
                        dynamic_cast<CrossEntropyWithLossLayer<Dtype>*>(lcDGAN->_lastLayers[0]);
                    SASSERT0(lossDRealLayer != NULL);

                    float realDRealAvg = 0.0;
                    const Dtype* dLossReal = lossDRealLayer->_outputData[0]->host_data();
                    for (int depth = 0; depth < ncDGAN->_batchSize; depth++) {
                        realDRealAvg += dLossReal[depth];
                    }
                    realDRealAvg /= (float)ncDGAN->_batchSize;

                    CrossEntropyWithLossLayer<Dtype>* lossDFakeLayer =
                        dynamic_cast<CrossEntropyWithLossLayer<Dtype>*>(lcGD0GAN->_lastLayers[0]);
                    SASSERT0(lossDFakeLayer != NULL);

                    float realDFakeAvg = 0.0;
                    const Dtype* dLossFake = lossDFakeLayer->_outputData[0]->host_data();
                    for (int depth = 0; depth < ncDGAN->_batchSize; depth++) {
                        realDFakeAvg += dLossFake[depth];
                    }
                    realDFakeAvg /= (float)ncDGAN->_batchSize;

                    if (j % 100 == 0)
                        cout << "LOSS D=" << realDRealAvg + realDFakeAvg << "(" <<
                            realDRealAvg << "," << realDFakeAvg << ")" << endl;
                }
            }

            CrossEntropyWithLossLayer<Dtype>* lossLayer =
                dynamic_cast<CrossEntropyWithLossLayer<Dtype>*>(lcGD0GAN->_lastLayers[0]);
            SASSERT0(lossLayer != NULL);
            NoiseInputLayer<Dtype>* noiseInputLayer =
                dynamic_cast<NoiseInputLayer<Dtype>*>(lcGD0GAN->_firstLayers[0]);
            SASSERT0(noiseInputLayer != NULL);

            lossLayer->setTargetValue(1.0);
            //noiseInputLayer->setRegenerateNoise(false);

            for (int loop = 0; loop < 2; loop++) {
                networkGD0GAN->_feedforward(0);
                networkGD0GAN->_backpropagation(0);
                // update
                //for (uint32_t k = 0; k < lcGD0GAN->_learnableLayers.size(); k++) {
                for (uint32_t k = 0; k < 9; k++) {
                    lcGD0GAN->_learnableLayers[k]->update();
                }

                CrossEntropyWithLossLayer<Dtype>* lossGFakeLayer =
                    dynamic_cast<CrossEntropyWithLossLayer<Dtype>*>(lcGD0GAN->_lastLayers[0]);
                SASSERT0(lossGFakeLayer != NULL);

                float realGFakeAvg = lossGFakeLayer->cost();
                if (j % 100 == 0)
                    cout << "LOSS G=" << realGFakeAvg << endl;
            }

            lossLayer->setTargetValue(0.0);
            noiseInputLayer->setRegenerateNoise(true);

            /*
            if (j % 100 == 0) {
                setLayerTrain(lcGD0GAN, false);

                char temp[64];
                sprintf(temp, "G(epoch=%d)", i);
        
                networkGD0GAN->_feedforward(0);
                //DebugUtil<Dtype>::printNetworkEdges(stdout, "GEpoch", lcGD0GAN, 0);

                Layer<Dtype>* convLayer = lcGD0GAN->_nameLayerMap["ConvLayer1"];
                const Dtype* host_data = convLayer->_inputData[0]->host_data();
                ImageUtil<Dtype>::saveImage(host_data, 64, 3, 64, 64, "");

                setLayerTrain(lcGD0GAN, true);
            }
            */
        }

        if (true) {
            setLayerTrain(lcGD0GAN, false);

            char temp[64];
            sprintf(temp, "G(epoch=%d)", i);
    
            networkGD0GAN->_feedforward(0);
            //DebugUtil<Dtype>::printNetworkEdges(stdout, "GEpoch", lcGD0GAN, 0);

            Layer<Dtype>* convLayer = lcGD0GAN->_nameLayerMap["ConvLayer1"];
            const Dtype* host_data = convLayer->_inputData[0]->host_data();
            ImageUtil<Dtype>::saveImage(host_data, 64, 3, 64, 64, "");

            setLayerTrain(lcGD0GAN, true);
        }
    }

    /*
    // noise check
    setLayerTrain(lcGD0GAN, false);

    NoiseInputLayer<Dtype>* noiseInputLayer =
        dynamic_cast<NoiseInputLayer<Dtype>*>(lcGD0GAN->_firstLayers[0]);
    SASSERT0(noiseInputLayer != NULL);
    noiseInputLayer->setRegenerateNoise(false);

    for (int i = 0; i < 100; i++) {
        float noise = -0.9;
        
        while (noise < 1.0) {
            string folderName = "noise_" + to_string(i) + "_" + to_string(noise);
            cout << "folderName : " << folderName << endl;

            Dtype* noiseData = noiseInputLayer->_inputData[0]->mutable_host_data();
            for (int j = 0; j < 100; j++) {
                if (i == j) {
                    noiseData[j] = noise;
                } else {
                    noiseData[j] = 0.001;
                }
            }

            networkGD0GAN->_feedforward(0);
            Layer<Dtype>* convLayer = lcGD0GAN->_nameLayerMap["ConvLayer1"];
            const Dtype* host_data = convLayer->_inputData[0]->host_data();
            ImageUtil<Dtype>::saveImage(host_data, 64, 3, 64, 64, folderName);

            noise += 0.1;
        }
    }
    */
#endif
}


template class GAN<float>;
