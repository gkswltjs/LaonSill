#if 1
#include <fcntl.h>
#include <cstdint>
#include <vector>
#include <iostream>
#include <fstream>

#include "cuda/Cuda.h"

#include "gnuplot-iostream.h"

#include "jsoncpp/json/json.h"

#include "common.h"
#include "DataSet.h"
#include "MockDataSet.h"
#include "Debug.h"
#include "Evaluation.h"
#include "Top1Evaluation.h"
#include "Top5Evaluation.h"
#include "NetworkMonitor.h"
#include "Network.h"
#include "NetworkConfig.h"
#include "Util.h"
#include "Worker.h"
#include "Job.h"
#include "Communicator.h"
#include "Client.h"
#include "InitParam.h"
#include "Param.h"
#include "ColdLog.h"
#include "SysLog.h"
#include "HotLog.h"
#include "StdOutLog.h"
#include "Perf.h"
#include "Atari.h"
#include "Broker.h"
#include "test.h"
#include "DQNImageLearner.h"
#include "ImageUtil.h"
#include "DebugUtil.h"

using namespace std;

#ifndef CLIENT_MODE

void printUsageAndExit(char* prog) {
    fprintf(stderr,
        "Usage: %s [-v] [-d | -f jobFilePath | -a romFilePath | -t testItemName]\n", prog);
    exit(EXIT_FAILURE);
}

// FIXME: 디버깅용 함수.... 나중에 싹다 지우자.
void setLayerTrain(LayersConfig<float>* lc, bool train) {
    int layerCount = lc->_layers.size();

    for (int i = 0; i < layerCount; i++) {
        BatchNormLayer<float>* bnLayer = dynamic_cast<BatchNormLayer<float>*>(lc->_layers[i]);

        if (bnLayer == NULL)
            continue;

        bnLayer->setTrain(train);
    }
}

typedef struct top10Sort_s {
    float value;
    int index;

    bool operator < (const struct top10Sort_s &x) const {
        return value < x.value;
    }
} top10Sort;

// XXX: inefficient..
int getTop10GuessSuccessCount(const float* data, const float* label, int batchCount,
    int depth, bool train, int epoch, const float* image, int imageBaseIndex,
    vector<EtriData> etriData) {

    int successCnt = 0;

#if 1
    string folderName;
        if (train) {
            folderName = "train_" + to_string(epoch) + "_" + to_string(imageBaseIndex); 
        } else {
            folderName = "test_" + to_string(epoch) + "_" + to_string(imageBaseIndex); 
        }
        
        ImageUtil<float>::saveImage(image, batchCount, 3, 224, 224, folderName);
#endif

    for (int i = 0; i < batchCount; i++) {
        vector<int> curLabel;
        vector<top10Sort> tempData;

        for (int j = 0; j < depth; j++) {
            int index = i * depth + j;

            if (label[index] > 0.99) {
                curLabel.push_back(j);
            }

            tempData.push_back({data[index], j});
        }

        sort(tempData.begin(), tempData.end());

        bool found = false;
        for (int j = 0; j < 10; j++) {
            int reverseIndex = depth - 1 - j;
            int target = tempData[reverseIndex].index;

            for (int k = 0; k < curLabel.size(); k++) {
                if (curLabel[k] == target) {
                    found = true;
                    break;
                }
            }

            if (found)
                break;
        }

#if 0
        printf ("Labels[%d] : ", i);
        for (int j = 0; j < curLabel.size(); j++) {
            printf(" %d", curLabel[j]);
        }
        printf ("\n");

        printf ("top 10 data[%d] : ", i);
        for (int j = 0; j < 10; j++) {
            int reverseIndex = depth - 1 - j;
            int target = tempData[reverseIndex].index;
            printf(" %d", target);
        }
        printf("\n");

        int imageIndex = i + imageBaseIndex;
        cout << "[folder:" << folderName << "] : " << etriData[imageIndex].filePath <<
            ", labels : ";
        for (int k = 0; k < etriData[imageIndex].labels.size(); k++) {
            cout << etriData[imageIndex].labels[k] << " ";
        }
        cout << endl;
#endif

        if (found)
            successCnt++;
    }

#if 0
    for (int i = 0; i < batchCount; i++) {
        printf("Labels[%d] : ", i);
        for (int j = 0; j < depth; j++) {
            int index = i * depth + j;
            printf("%d ", label[index]);
        }
        printf("\n");
        printf("data[%d] : ", i);
        for (int j = 0; j < depth; j++) {
            int index = i * depth + j;
            printf("%d ", data[index]);
        }
        printf("\n");
    }
#endif

    return successCnt;
}


void runGAN() {

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

	NetworkConfig<float>* ncDGAN =
			(new typename NetworkConfig<float>::Builder())
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

	NetworkConfig<float>* ncGD0GAN =
			(new typename NetworkConfig<float>::Builder())
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

 	Network<float>* networkDGAN = new Network<float>(ncDGAN);
 	Network<float>* networkGD0GAN = new Network<float>(ncGD0GAN);

    // (1) layer config를 만든다. 이 과정중에 layer들의 초기화가 진행된다.
	LayersConfig<float>* lcGD0GAN = createGD0OfGANLayersConfig<float>();
	LayersConfig<float>* lcDGAN = createDOfGANLayersConfig<float>();
 	networkGD0GAN->setLayersConfig(lcGD0GAN);
 	networkDGAN->setLayersConfig(lcDGAN);

	// (2) network config 정보를 layer들에게 전달한다.
	for (uint32_t i = 0; i < lcDGAN->_layers.size(); i++)
		lcDGAN->_layers[i]->setNetworkConfig(ncDGAN);

	for (uint32_t i = 0; i < lcGD0GAN->_layers.size(); i++)
		lcGD0GAN->_layers[i]->setNetworkConfig(ncGD0GAN);

    for (int i = 0; i < 100000; i++) {  // epoch
        cout << "epoch=" << i << endl;

        InputLayer<float>* inputLayer = lcDGAN->_inputLayer;
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
                    CrossEntropyWithLossLayer<float>* lossDRealLayer =
                        dynamic_cast<CrossEntropyWithLossLayer<float>*>(lcDGAN->_lastLayers[0]);
                    SASSERT0(lossDRealLayer != NULL);

                    float realDRealAvg = 0.0;
                    const float* dLossReal = lossDRealLayer->_outputData[0]->host_data();
                    for (int depth = 0; depth < ncDGAN->_batchSize; depth++) {
                        realDRealAvg += dLossReal[depth];
                    }
                    realDRealAvg /= (float)ncDGAN->_batchSize;

                    CrossEntropyWithLossLayer<float>* lossDFakeLayer =
                        dynamic_cast<CrossEntropyWithLossLayer<float>*>(lcGD0GAN->_lastLayers[0]);
                    SASSERT0(lossDFakeLayer != NULL);

                    float realDFakeAvg = 0.0;
                    const float* dLossFake = lossDFakeLayer->_outputData[0]->host_data();
                    for (int depth = 0; depth < ncDGAN->_batchSize; depth++) {
                        realDFakeAvg += dLossFake[depth];
                    }
                    realDFakeAvg /= (float)ncDGAN->_batchSize;

                    if (j % 100 == 0)
                        cout << "LOSS D=" << realDRealAvg + realDFakeAvg << "(" <<
                            realDRealAvg << "," << realDFakeAvg << ")" << endl;
                }
            }

            CrossEntropyWithLossLayer<float>* lossLayer =
                dynamic_cast<CrossEntropyWithLossLayer<float>*>(lcGD0GAN->_lastLayers[0]);
            SASSERT0(lossLayer != NULL);
            NoiseInputLayer<float>* noiseInputLayer =
                dynamic_cast<NoiseInputLayer<float>*>(lcGD0GAN->_firstLayers[0]);
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

                CrossEntropyWithLossLayer<float>* lossGFakeLayer =
                    dynamic_cast<CrossEntropyWithLossLayer<float>*>(lcGD0GAN->_lastLayers[0]);
                SASSERT0(lossGFakeLayer != NULL);

                float realGFakeAvg = lossGFakeLayer->cost();
                if (j % 100 == 0)
                    cout << "LOSS G=" << realGFakeAvg << endl;
            }

            lossLayer->setTargetValue(0.0);
            noiseInputLayer->setRegenerateNoise(true);

#if 0
            if (j % 100 == 0) {
                setLayerTrain(lcGD0GAN, false);

                char temp[64];
                sprintf(temp, "G(epoch=%d)", i);
        
                networkGD0GAN->_feedforward(0);
                //DebugUtil<float>::printNetworkEdges(stdout, "GEpoch", lcGD0GAN, 0);

                Layer<float>* convLayer = lcGD0GAN->_nameLayerMap["ConvLayer1"];
                const float* host_data = convLayer->_inputData[0]->host_data();
                ImageUtil<float>::saveImage(host_data, 64, 3, 64, 64, "");

                setLayerTrain(lcGD0GAN, true);
            }
#endif
        }

        if (true) {
            setLayerTrain(lcGD0GAN, false);

            char temp[64];
            sprintf(temp, "G(epoch=%d)", i);
    
            networkGD0GAN->_feedforward(0);
            //DebugUtil<float>::printNetworkEdges(stdout, "GEpoch", lcGD0GAN, 0);

            Layer<float>* convLayer = lcGD0GAN->_nameLayerMap["ConvLayer1"];
            const float* host_data = convLayer->_inputData[0]->host_data();
            ImageUtil<float>::saveImage(host_data, 64, 3, 64, 64, "");

            setLayerTrain(lcGD0GAN, true);
        }
    }

#if 0
    // noise check
    setLayerTrain(lcGD0GAN, false);

    NoiseInputLayer<float>* noiseInputLayer =
        dynamic_cast<NoiseInputLayer<float>*>(lcGD0GAN->_firstLayers[0]);
    SASSERT0(noiseInputLayer != NULL);
    noiseInputLayer->setRegenerateNoise(false);

    for (int i = 0; i < 100; i++) {
        float noise = -0.9;
        
        while (noise < 1.0) {
            string folderName = "noise_" + to_string(i) + "_" + to_string(noise);
            cout << "folderName : " << folderName << endl;

            float* noiseData = noiseInputLayer->_inputData[0]->mutable_host_data();
            for (int j = 0; j < 100; j++) {
                if (i == j) {
                    noiseData[j] = noise;
                } else {
                    noiseData[j] = 0.001;
                }
            }

            networkGD0GAN->_feedforward(0);
            Layer<float>* convLayer = lcGD0GAN->_nameLayerMap["ConvLayer1"];
            const float* host_data = convLayer->_inputData[0]->host_data();
            ImageUtil<float>::saveImage(host_data, 64, 3, 64, 64, folderName);

            noise += 0.1;
        }
    }
#endif

}

void runYolo() {
    // loss layer of Discriminator GAN 
	const vector<string> lossList = { "celossEtri" };
    // loss layer of Generatoer-Discriminator 0 GAN

	const NetworkPhase phase = NetworkPhase::TrainPhase;
	const uint32_t batchSize = 16;
	const uint32_t testInterval = 1;		// 10000(목표 샘플수) / batchSize
	const uint32_t saveInterval = 100000;		// 1000000 / batchSize
	const float baseLearningRate = 0.001f;

	const uint32_t stepSize = 100000;
	const float weightDecay = 0.0001f;
	const float momentum = 0.9f;
	const float clipGradientsLevel = 0.0f;
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

	NetworkConfig<float>* networkConfig =
			(new typename NetworkConfig<float>::Builder())
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
				new NetworkMonitor("celossEtri", NetworkMonitor::PLOT_ONLY),
				})
			->lossLayers(lossList)
            ->optimizer(opt)
			->build();

	Util::printVramInfo();

 	Network<float>* network = new Network<float>(networkConfig);

    // (1) layer config를 만든다. 이 과정중에 layer들의 초기화가 진행된다.
	LayersConfig<float>* layersConfig = createEtriVGG19NetLayersConfig<float>();
 	network->setLayersConfig(layersConfig);

	// (2) network config 정보를 layer들에게 전달한다.
	for (uint32_t i = 0; i < layersConfig->_layers.size(); i++)
		layersConfig->_layers[i]->setNetworkConfig(networkConfig);


    network->sgd(3);
}

void runEtri() {
    // loss layer of Discriminator GAN 
	const vector<string> lossList = { "celossEtri" };
    // loss layer of Generatoer-Discriminator 0 GAN

	const NetworkPhase phase = NetworkPhase::TrainPhase;
	const uint32_t batchSize = 16;
	const uint32_t testInterval = 1;		// 10000(목표 샘플수) / batchSize
	const uint32_t saveInterval = 100000;		// 1000000 / batchSize
	const float baseLearningRate = 0.001f;

	const uint32_t stepSize = 100000;
	const float weightDecay = 0.0001f;
	const float momentum = 0.9f;
	const float clipGradientsLevel = 0.0f;
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

	NetworkConfig<float>* networkConfig =
			(new typename NetworkConfig<float>::Builder())
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
				new NetworkMonitor("celossEtri", NetworkMonitor::PLOT_ONLY),
				})
			->lossLayers(lossList)
            ->optimizer(opt)
			->build();

	Util::printVramInfo();

 	Network<float>* network = new Network<float>(networkConfig);

    // (1) layer config를 만든다. 이 과정중에 layer들의 초기화가 진행된다.
	LayersConfig<float>* layersConfig = createEtriVGG19NetLayersConfig<float>();
 	network->setLayersConfig(layersConfig);

	// (2) network config 정보를 layer들에게 전달한다.
	for (uint32_t i = 0; i < layersConfig->_layers.size(); i++)
		layersConfig->_layers[i]->setNetworkConfig(networkConfig);

    // (3) 학습한다.
    for (int epoch = 0; epoch < 50; epoch++) {
        STDOUT_BLOCK(cout << "epoch #" << epoch << " starts" << endl;); 

        EtriInputLayer<float>* etriInputLayer =
            dynamic_cast<EtriInputLayer<float>*>(layersConfig->_firstLayers[0]);
        SASSERT0(etriInputLayer != NULL);

        CrossEntropyWithLossLayer<float>* lossLayer =
            dynamic_cast<CrossEntropyWithLossLayer<float>*>(layersConfig->_lastLayers[0]);
        SASSERT0(lossLayer != NULL);

        const uint32_t trainDataSize = etriInputLayer->getNumTrainData();
        const uint32_t numTrainBatches = trainDataSize / networkConfig->_batchSize - 1;

        // (3-1) 네트워크를 학습한다.
        for (int i = 0; i < numTrainBatches; i++) {
            STDOUT_BLOCK(cout << "train data(" << i << "/" << numTrainBatches << ")" <<
                endl;);
            network->sgdMiniBatch(i);
        }

        // (3-2) 트레이닝 데이터에 대한 평균 Loss와 정확도를 구한다.
        STDOUT_BLOCK(cout << "evaluate train data(num train batches =" << numTrainBatches <<
            ")" << endl;);
        float trainLoss = 0.0;
        int trainSuccessCnt = 0;
        for (int i = 0; i < numTrainBatches; i++) {
            network->_feedforward(i);
            trainLoss += lossLayer->cost();

            const float* inputData = etriInputLayer->_inputData[0]->host_data();
            const float* outputData = lossLayer->_inputData[0]->host_data();
            const float* outputLabel = lossLayer->_inputData[1]->host_data();
            trainSuccessCnt += getTop10GuessSuccessCount(outputData, outputLabel,
                networkConfig->_batchSize, 1000, true, epoch, inputData,
                (int)(networkConfig->_batchSize * i), etriInputLayer->trainData);
        }
        trainLoss = trainLoss / (float)(numTrainBatches);

        // (3-3) 테스트 데이터에 대한 평균 Loss와 정확도를 구한다.
        etriInputLayer->setTrain(false);

        const uint32_t testDataSize = etriInputLayer->getNumTestData();
        const uint32_t numTestBatches = testDataSize / networkConfig->_batchSize - 1;

        STDOUT_BLOCK(cout << "evaluate test data(num test batches =" << numTestBatches <<
            ")" << endl;);
        float testLoss = 0.0;
        int testSuccessCnt = 0;
        for (int i = 0; i < numTestBatches; i++) {
            network->_feedforward(i);
            testLoss += lossLayer->cost();

            const float* inputData = etriInputLayer->_inputData[0]->host_data();
            const float* outputData = lossLayer->_inputData[0]->host_data();
            const float* outputLabel = lossLayer->_inputData[1]->host_data();
            testSuccessCnt += getTop10GuessSuccessCount(outputData, outputLabel,
                networkConfig->_batchSize, 1000, false, epoch, inputData,
                (int)(networkConfig->_batchSize * i), etriInputLayer->testData);
        }
        testLoss = testLoss / (float)(numTestBatches);

        etriInputLayer->setTrain(true);

        float trainAcc = (float)trainSuccessCnt / (float)numTrainBatches /
            (float)networkConfig->_batchSize;
        float testAcc = (float)testSuccessCnt / (float)numTestBatches /
            (float)networkConfig->_batchSize;
        STDOUT_BLOCK(cout << "[RESULT #" << epoch << "] train loss : " << trainLoss <<
            ", test losss : " << testLoss << ", train accuracy : " << trainAcc << "(" <<
            trainSuccessCnt << "/" << numTrainBatches * networkConfig->_batchSize <<
            "), test accuracy : " << testAcc << "(" << testSuccessCnt << "/" <<
            numTestBatches * networkConfig->_batchSize << ")" << endl;);
    }

#if 0
    network->_feedforward(0);
    DebugUtil<float>::printNetworkEdges(stdout, "etri", layersConfig, 0);
    network->_backpropagation(0);
    DebugUtil<float>::printNetworkEdges(stdout, "etri", layersConfig, 0);

    for (uint32_t i = 0; i < layersConfig->_learnableLayers.size(); i++) {
        layersConfig->_learnableLayers[i]->update();
    }
    DebugUtil<float>::printNetworkEdges(stdout, "etri", layersConfig, 0);
#endif
}

void developerMain() {
    STDOUT_LOG("enter developerMain()");

    checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cublasCreate(&Cuda::cublasHandle));
	checkCUDNN(cudnnCreate(&Cuda::cudnnHandle));

    //runGAN();
    //runEtri();
    runYolo();

    STDOUT_LOG("exit developerMain()");
}

void loadJobFile(const char* fileName, Json::Value& rootValue) {
    filebuf fb;
    if (fb.open(fileName, ios::in) == NULL) {
        fprintf(stderr, "ERROR: cannot open %s\n", fileName);
        exit(EXIT_FAILURE);
    }

    istream is(&fb);
    Json::Reader reader;
    bool parse = reader.parse(is, rootValue);

    if (!parse) {
        fprintf(stderr, "ERROR: invalid json-format file.\n");
        fprintf(stderr, "%s\n", reader.getFormattedErrorMessages().c_str());
        fb.close();
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
    int     opt;


    // 처음 생각했던 것보다 실행모드의 개수가 늘었다.
    // 모드가 하나만 더 추가되면 그냥 enum type으로 모드를 정의하도록 하자.

    bool    useDeveloperMode = false; 
    bool    useSingleJobMode = false;
    bool    useRLMode = false;
    bool    useTestMode = false;

    char*   singleJobFilePath;
    char*   romFilePath;
    char*   testItemName;

    // (1) 옵션을 읽는다.
    while ((opt = getopt(argc, argv, "vdf:a:t:")) != -1) {
        switch (opt) {
        case 'v':
            printf("%s version %d.%d.%d\n", argv[0], SPARAM(VERSION_MAJOR),
                SPARAM(VERSION_MINOR), SPARAM(VERSION_PATCH));
            exit(EXIT_SUCCESS);

        case 'd':
            if (useSingleJobMode | useRLMode | useTestMode)
                printUsageAndExit(argv[0]);
            useDeveloperMode = true;
            break;

        case 'f':
            if (useDeveloperMode | useRLMode | useTestMode)
                printUsageAndExit(argv[0]);
            useSingleJobMode = true;
            singleJobFilePath = optarg;
            break;

        case 'a':
            if (useDeveloperMode | useSingleJobMode | useTestMode)
                printUsageAndExit(argv[0]);
            useRLMode = true;
            romFilePath = optarg;
            break;

        case 't':
            if (useSingleJobMode | useDeveloperMode | useRLMode)
                printUsageAndExit(argv[0]);
            useTestMode = true;
            testItemName = optarg;
            checkTestItem(testItemName);
            break;

        default:    /* ? */
            printUsageAndExit(argv[0]);
            break; 
        }
    }

    // COMMENT: 만약 이후에 인자를 받고 싶다면 optind를 기준으로 인자를 받으면 된다.
    //  ex. Usage: %s [-d | -f jobFilePath] hostPort 와 같은 식이라면
    //  hostPort = atoi(argv[optind]);로 인자값을 받으면 된다.
    //  개인적으로 host port와 같은 정보는 SPARAM으로 정의하는 것을 더 선호한다.

    // (2) 서버 시작 시간 측정을 시작한다.
    struct timespec startTime;
    SPERF_START(SERVER_RUNNING_TIME, &startTime);
	STDOUT_BLOCK(cout << "SOOOA engine starts" << endl;);

    // (3) 파라미터, 로깅, job 모듈을 초기화 한다.
    InitParam::init();
    Perf::init();
    SysLog::init();
    ColdLog::init();
    Job::init();
    Broker::init();
    Network<float>::init();
    DQNImageLearner<float>::init();

    if (!useDeveloperMode) {
        HotLog::init();
    	HotLog::launchThread(SPARAM(CONSUMER_COUNT) + 1);
    }
    SYS_LOG("Logging system is initialized...");

    // (4) 뉴럴 네트워크 관련 기본 설정을 한다.
    //     TODO: SPARAM의 인자로 대체하자.
	cout.precision(7);
	cout.setf(ios::fixed);
	Util::setOutstream(&cout);
	Util::setPrint(false);

    // (5) 모드에 따른 동작을 수행한다.
    if (useDeveloperMode) {
        // (5-A-1) Cuda를 생성한다.
        Cuda::create(SPARAM(GPU_COUNT));
        COLD_LOG(ColdLog::INFO, true, "CUDA is initialized");

        // (5-A-2) DeveloperMain()함수를 호출한다.
        developerMain();

        // (5-A-3) 자원을 해제 한다.
        Cuda::destroy();
    } else if (useSingleJobMode) {
        // FIXME: we do not support this mode until declaration of job type is done
#if 0
        // TODO: 아직 만들다 말았음
        // (5-B-1) Job File(JSON format)을 로딩한다.
        Json::Value rootValue;
        loadJobFile(singleJobFilePath, rootValue);

        // (5-B-2) Producer&Consumer를 생성.
        Worker<float>::launchThreads(SPARAM(CONSUMER_COUNT));
        
        // (5-B-3) Network를 생성한다.
        // TODO: Network configuration에 대한 정의 필요
        // XXX: 1개의 Network만 있다고 가정하고 있음.
        string networkConf = rootValue.get("Network", "").asString();
        int networkId = Worker<float>::createNetwork();
        Network<float>* network = Worker<float>::getNetwork(networkId); 
        SASSUME0(network);
        
        // (5-B-4) Job을 생성한다.
        // TODO: Job configuration에 대한 정의 필요
        Json::Value jobConfList = rootValue["Job"];
        for (int jobIndex = 0; jobIndex < jobConfList.size(); jobIndex++) {
            Json::Value jobConf = jobConfList[jobIndex];
            SASSUME0(jobConf.size() == 2);
            
            Job* newJob = new Job((Job::JobType)(jobConf[0].asInt()), network,
                                (jobConf[1].asInt()));
            Worker<float>::pushJob(newJob);
        }
#endif
        
        // (5-B-5) Worker Thread (Producer& Consumer)를 종료를 요청한다.
        Job* haltJob = new Job(Job::HaltMachine);
        Worker<float>::pushJob(haltJob);

        // (5-B-6) Producer&Consumer를 종료를 기다린다.
        Worker<float>::joinThreads();
    } else if (useRLMode) {
        // (5-C-1) Producer&Consumer를 생성.
        Worker<float>::launchThreads(SPARAM(CONSUMER_COUNT));

        // (5-C-3) Layer를 생성한다.
        Atari::run(romFilePath);

        Worker<float>::joinThreads();
    } else if (useTestMode) {
        // (5-D-1) Producer&Consumer를 생성.
        Worker<float>::launchThreads(SPARAM(CONSUMER_COUNT));

        // (5-D-2) Listener & Sess threads를 생성.
        Communicator::launchThreads(SPARAM(SESS_COUNT));

        // (5-D-3) 테스트를 실행한다.
        runTest(testItemName);

        // (5-D-4) release resources 
        Job* haltJob = new Job(Job::HaltMachine);
        Worker<float>::pushJob(haltJob);

        Communicator::halt();       // threads will be eventually halt

        // (5-D-5) 각각의 쓰레드들의 종료를 기다린다.
        Worker<float>::joinThreads();
        Communicator::joinThreads();
    } else {
        // (5-E-1) Producer&Consumer를 생성.
        Worker<float>::launchThreads(SPARAM(CONSUMER_COUNT));

        // (5-E-2) Listener & Sess threads를 생성.
        Communicator::launchThreads(SPARAM(SESS_COUNT));

        // (5-E-3) 각각의 쓰레드들의 종료를 기다린다.
        Worker<float>::joinThreads();
        Communicator::joinThreads();
    }

    // (6) 로깅 관련 모듈이 점유했던 자원을 해제한다.
    if (!useDeveloperMode)
        HotLog::destroy();
    ColdLog::destroy();
    SysLog::destroy();
    Broker::destroy();

    // (7) 서버 종료 시간을 측정하고, 계산하여 서버 실행 시간을 출력한다.
    SPERF_END(SERVER_RUNNING_TIME, startTime);
    STDOUT_LOG("server running time : %lf\n", SPERF_TIME(SERVER_RUNNING_TIME));
	STDOUT_BLOCK(cout << "SOOOA engine ends" << endl;);

    InitParam::destroy();
	exit(EXIT_SUCCESS);
}



#else

const char          SERVER_HOSTNAME[] = {"localhost"};
int main(int argc, char** argv) {
    Client::clientMain(SERVER_HOSTNAME, Communicator::LISTENER_PORT);
	exit(EXIT_SUCCESS);
}
#endif
#endif

