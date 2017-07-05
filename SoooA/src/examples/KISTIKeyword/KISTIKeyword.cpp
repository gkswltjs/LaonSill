/**
 * @file KISTIKeyword.cpp
 * @date 2017-04-20
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <vector>

#include "common.h"
#include "Debug.h"
#include "KISTIKeyword.h"
#include "StdOutLog.h"
#include "Network.h"
#include "NetworkMonitor.h"
#include "ImageUtil.h"
#include "PlanParser.h"
#include "PropMgmt.h"

using namespace std;

#define EXAMPLE_KISTIKEYWORD_NETWORK_FILEPATH   ("../src/examples/KISTIKeyword/network.json")

#if 0
// XXX: inefficient..
template<typename Dtype>
int KISTIKeyword<Dtype>::getTop10GuessSuccessCount(const float* data,
    const float* label, int batchCount, int depth, bool train, int epoch, 
    const float* image, int imageBaseIndex, vector<KistiData> etriData) {

#if 0
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
#else
    return -1;
#endif
}


template <typename Dtype>
LayersConfig<Dtype>* KISTIKeyword<Dtype>::createKistiVGG19NetLayersConfig() {
#if 0
    const float bias_const = 0.2f;

    LayersConfig<Dtype>* layersConfig = (new typename LayersConfig<Dtype>::Builder())
            ->layer((new typename KistiInputLayer<Dtype>::Builder())
                    ->id(0)
                    ->name("data")
                    ->image(std::string(SPARAM(BASE_DATA_DIR)) +
                            std::string("/etri/flatten/"))
                    ->outputs({"data", "label"}))

            // tier 1
            ->layer((new typename ConvLayer<Dtype>::Builder())
                    ->id(1)
                    ->name("conv1_1")
                    ->filterDim(3, 3, 3, 64, 1, 1)
                    ->weightUpdateParam(1, 1)
                    ->biasUpdateParam(2, 0)
                    ->weightFiller(ParamFillerType::Xavier, 0.1)
                    ->biasFiller(ParamFillerType::Constant, bias_const)
                    ->inputs({"data"})
                    ->outputs({"conv1_1"}))

            ->layer((new typename ReluLayer<Dtype>::Builder())
                    ->id(2)
                    ->name("relu1_1")
                    ->inputs({"conv1_1"})
                    ->outputs({"conv1_1"}))

            ->layer((new typename ConvLayer<Dtype>::Builder())
                    ->id(3)
                    ->name("conv1_2")
                    ->filterDim(3, 3, 64, 64, 1, 1)
                    ->weightUpdateParam(1, 1)
                    ->biasUpdateParam(2, 0)
                    ->weightFiller(ParamFillerType::Xavier, 0.1)
                    ->biasFiller(ParamFillerType::Constant, bias_const)
                    ->inputs({"conv1_1"})
                    ->outputs({"conv1_2"}))

            ->layer((new typename ReluLayer<Dtype>::Builder())
                    ->id(4)
                    ->name("relu1_2")
                    ->inputs({"conv1_2"})
                    ->outputs({"conv1_2"}))

            ->layer((new typename PoolingLayer<Dtype>::Builder())
                    ->id(5)
                    ->name("pool1")
                    ->poolDim(2, 2, 0, 2)
                    ->poolingType(PoolingType::Max)
                    ->inputs({"conv1_2"})
                    ->outputs({"pool1"}))

            // tier 2
            ->layer((new typename ConvLayer<Dtype>::Builder())
                    ->id(6)
                    ->name("conv2_1")
                    ->filterDim(3, 3, 64, 128, 1, 1)
                    ->weightUpdateParam(1, 1)
                    ->biasUpdateParam(2, 0)
                    ->weightFiller(ParamFillerType::Xavier, 0.1)
                    ->biasFiller(ParamFillerType::Constant, bias_const)
                    ->inputs({"pool1"})
                    ->outputs({"conv2_1"}))

            ->layer((new typename ReluLayer<Dtype>::Builder())
                    ->id(7)
                    ->name("relu2_1")
                    ->inputs({"conv2_1"})
                    ->outputs({"conv2_1"}))

            ->layer((new typename ConvLayer<Dtype>::Builder())
                    ->id(8)
                    ->name("conv2_2")
                    ->filterDim(3, 3, 128, 128, 1, 1)
                    ->weightUpdateParam(1, 1)
                    ->biasUpdateParam(2, 0)
                    ->weightFiller(ParamFillerType::Xavier, 0.1)
                    ->biasFiller(ParamFillerType::Constant, bias_const)
                    ->inputs({"conv2_1"})
                    ->outputs({"conv2_2"}))

            ->layer((new typename ReluLayer<Dtype>::Builder())
                    ->id(9)
                    ->name("relu2_2")
                    ->inputs({"conv2_2"})
                    ->outputs({"conv2_2"}))

            ->layer((new typename PoolingLayer<Dtype>::Builder())
                    ->id(10)
                    ->name("pool2")
                    ->poolDim(2, 2, 0, 2)
                    ->poolingType(PoolingType::Max)
                    ->inputs({"conv2_2"})
                    ->outputs({"pool2"}))

            // tier 3
            ->layer((new typename ConvLayer<Dtype>::Builder())
                    ->id(11)
                    ->name("conv3_1")
                    ->filterDim(3, 3, 128, 256, 1, 1)
                    ->weightUpdateParam(1, 1)
                    ->biasUpdateParam(2, 0)
                    ->weightFiller(ParamFillerType::Xavier, 0.1)
                    ->biasFiller(ParamFillerType::Constant, bias_const)
                    ->inputs({"pool2"})
                    ->outputs({"conv3_1"}))

            ->layer((new typename ReluLayer<Dtype>::Builder())
                    ->id(12)
                    ->name("relu3_1")
                    ->inputs({"conv3_1"})
                    ->outputs({"conv3_1"}))

            ->layer((new typename ConvLayer<Dtype>::Builder())
                    ->id(13)
                    ->name("conv3_2")
                    ->filterDim(3, 3, 256, 256, 1, 1)
                    ->weightUpdateParam(1, 1)
                    ->biasUpdateParam(2, 0)
                    ->weightFiller(ParamFillerType::Xavier, 0.1)
                    ->biasFiller(ParamFillerType::Constant, bias_const)
                    ->inputs({"conv3_1"})
                    ->outputs({"conv3_2"}))

            ->layer((new typename ReluLayer<Dtype>::Builder())
                    ->id(14)
                    ->name("relu3_2")
                    ->inputs({"conv3_2"})
                    ->outputs({"conv3_2"}))

            ->layer((new typename ConvLayer<Dtype>::Builder())
                    ->id(15)
                    ->name("conv3_3")
                    ->filterDim(3, 3, 256, 256, 1, 1)
                    ->weightUpdateParam(1, 1)
                    ->biasUpdateParam(2, 0)
                    ->weightFiller(ParamFillerType::Xavier, 0.1)
                    ->biasFiller(ParamFillerType::Constant, bias_const)
                    ->inputs({"conv3_2"})
                    ->outputs({"conv3_3"}))

            ->layer((new typename ReluLayer<Dtype>::Builder())
                    ->id(16)
                    ->name("relu3_3")
                    ->inputs({"conv3_3"})
                    ->outputs({"conv3_3"}))

            ->layer((new typename ConvLayer<Dtype>::Builder())
                    ->id(17)
                    ->name("conv3_4")
                    ->filterDim(3, 3, 256, 256, 1, 1)
                    ->weightUpdateParam(1, 1)
                    ->biasUpdateParam(2, 0)
                    ->weightFiller(ParamFillerType::Xavier, 0.1)
                    ->biasFiller(ParamFillerType::Constant, bias_const)
                    ->inputs({"conv3_3"})
                    ->outputs({"conv3_4"}))

            ->layer((new typename ReluLayer<Dtype>::Builder())
                    ->id(18)
                    ->name("relu3_4")
                    ->inputs({"conv3_4"})
                    ->outputs({"conv3_4"}))

            ->layer((new typename PoolingLayer<Dtype>::Builder())
                    ->id(19)
                    ->name("pool3")
                    ->poolDim(2, 2, 0, 2)
                    ->poolingType(PoolingType::Max)
                    ->inputs({"conv3_4"})
                    ->outputs({"pool3"}))

            // tier 4
            ->layer((new typename ConvLayer<Dtype>::Builder())
                    ->id(20)
                    ->name("conv4_1")
                    ->filterDim(3, 3, 256, 512, 1, 1)
                    ->weightUpdateParam(1, 1)
                    ->biasUpdateParam(2, 0)
                    ->weightFiller(ParamFillerType::Xavier, 0.1)
                    ->biasFiller(ParamFillerType::Constant, bias_const)
                    ->inputs({"pool3"})
                    ->outputs({"conv4_1"}))

            ->layer((new typename ReluLayer<Dtype>::Builder())
                    ->id(21)
                    ->name("relu4_1")
                    ->inputs({"conv4_1"})
                    ->outputs({"conv4_1"}))

            ->layer((new typename ConvLayer<Dtype>::Builder())
                    ->id(22)
                    ->name("conv4_2")
                    ->filterDim(3, 3, 512, 512, 1, 1)
                    ->weightUpdateParam(1, 1)
                    ->biasUpdateParam(2, 0)
                    ->weightFiller(ParamFillerType::Xavier, 0.1)
                    ->biasFiller(ParamFillerType::Constant, bias_const)
                    ->inputs({"conv4_1"})
                    ->outputs({"conv4_2"}))

            ->layer((new typename ReluLayer<Dtype>::Builder())
                    ->id(23)
                    ->name("relu4_2")
                    ->inputs({"conv4_2"})
                    ->outputs({"conv4_2"}))

            ->layer((new typename ConvLayer<Dtype>::Builder())
                    ->id(24)
                    ->name("conv4_3")
                    ->filterDim(3, 3, 512, 512, 1, 1)
                    ->weightUpdateParam(1, 1)
                    ->biasUpdateParam(2, 0)
                    ->weightFiller(ParamFillerType::Xavier, 0.1)
                    ->biasFiller(ParamFillerType::Constant, bias_const)
                    ->inputs({"conv4_2"})
                    ->outputs({"conv4_3"}))

            ->layer((new typename ReluLayer<Dtype>::Builder())
                    ->id(25)
                    ->name("relu4_3")
                    ->inputs({"conv4_3"})
                    ->outputs({"conv4_3"}))

            ->layer((new typename ConvLayer<Dtype>::Builder())
                    ->id(26)
                    ->name("conv4_4")
                    ->filterDim(3, 3, 512, 512, 1, 1)
                    ->weightUpdateParam(1, 1)
                    ->biasUpdateParam(2, 0)
                    ->weightFiller(ParamFillerType::Xavier, 0.1)
                    ->biasFiller(ParamFillerType::Constant, bias_const)
                    ->inputs({"conv4_3"})
                    ->outputs({"conv4_4"}))

            ->layer((new typename ReluLayer<Dtype>::Builder())
                    ->id(27)
                    ->name("relu4_4")
                    ->inputs({"conv4_4"})
                    ->outputs({"conv4_4"}))

            ->layer((new typename PoolingLayer<Dtype>::Builder())
                    ->id(28)
                    ->name("pool4")
                    ->poolDim(2, 2, 0, 2)
                    ->poolingType(PoolingType::Max)
                    ->inputs({"conv4_4"})
                    ->outputs({"pool4"}))

            // tier 5
            ->layer((new typename ConvLayer<Dtype>::Builder())
                    ->id(29)
                    ->name("conv5_1")
                    ->filterDim(3, 3, 512, 512, 1, 1)
                    ->weightUpdateParam(1, 1)
                    ->biasUpdateParam(2, 0)
                    ->weightFiller(ParamFillerType::Xavier, 0.1)
                    ->biasFiller(ParamFillerType::Constant, bias_const)
                    ->inputs({"pool4"})
                    ->outputs({"conv5_1"}))

            ->layer((new typename ReluLayer<Dtype>::Builder())
                    ->id(30)
                    ->name("relu5_1")
                    ->inputs({"conv5_1"})
                    ->outputs({"conv5_1"}))

            ->layer((new typename ConvLayer<Dtype>::Builder())
                    ->id(31)
                    ->name("conv5_2")
                    ->filterDim(3, 3, 512, 512, 1, 1)
                    ->weightUpdateParam(1, 1)
                    ->biasUpdateParam(2, 0)
                    ->weightFiller(ParamFillerType::Xavier, 0.1)
                    ->biasFiller(ParamFillerType::Constant, bias_const)
                    ->inputs({"conv5_1"})
                    ->outputs({"conv5_2"}))

            ->layer((new typename ReluLayer<Dtype>::Builder())
                    ->id(32)
                    ->name("relu5_2")
                    ->inputs({"conv5_2"})
                    ->outputs({"conv5_2"}))

            ->layer((new typename ConvLayer<Dtype>::Builder())
                    ->id(33)
                    ->name("conv5_3")
                    ->filterDim(3, 3, 512, 512, 1, 1)
                    ->weightUpdateParam(1, 1)
                    ->biasUpdateParam(2, 0)
                    ->weightFiller(ParamFillerType::Xavier, 0.1)
                    ->biasFiller(ParamFillerType::Constant, bias_const)
                    ->inputs({"conv5_2"})
                    ->outputs({"conv5_3"}))

            ->layer((new typename ReluLayer<Dtype>::Builder())
                    ->id(34)
                    ->name("relu5_3")
                    ->inputs({"conv5_3"})
                    ->outputs({"conv5_3"}))

            ->layer((new typename ConvLayer<Dtype>::Builder())
                    ->id(35)
                    ->name("conv5_4")
                    ->filterDim(3, 3, 512, 512, 1, 1)
                    ->weightUpdateParam(1, 1)
                    ->biasUpdateParam(2, 0)
                    ->weightFiller(ParamFillerType::Xavier, 0.1)
                    ->biasFiller(ParamFillerType::Constant, bias_const)
                    ->inputs({"conv5_3"})
                    ->outputs({"conv5_4"}))

            ->layer((new typename ReluLayer<Dtype>::Builder())
                    ->id(36)
                    ->name("relu5_4")
                    ->inputs({"conv5_4"})
                    ->outputs({"conv5_4"}))

            ->layer((new typename PoolingLayer<Dtype>::Builder())
                    ->id(37)
                    ->name("pool5")
                    ->poolDim(2, 2, 0, 2)
                    ->poolingType(PoolingType::Max)
                    ->inputs({"conv5_4"})
                    ->outputs({"pool5"}))


            // classifier
            ->layer((new typename FullyConnectedLayer<Dtype>::Builder())
                    ->id(39)
                    ->name("fc6")
                    ->nOut(4096)
                    ->pDropout(0.5)
                    ->weightUpdateParam(1, 1)
                    ->biasUpdateParam(2, 0)
                    ->weightFiller(ParamFillerType::Xavier, 0.1)
                    ->biasFiller(ParamFillerType::Constant, bias_const)
                    ->inputs({"pool5"})
                    ->outputs({"fc6"}))

            ->layer((new typename ReluLayer<Dtype>::Builder())
                    ->id(40)
                    ->name("relu6")
                    ->inputs({"fc6"})
                    ->outputs({"fc6"}))

            ->layer((new typename FullyConnectedLayer<Dtype>::Builder())
                    ->id(41)
                    ->name("fc7")
                    ->nOut(4096)
                    ->pDropout(0.5)
                    ->weightUpdateParam(1, 1)
                    ->biasUpdateParam(2, 0)
                    ->weightFiller(ParamFillerType::Xavier, 0.1)
                    ->biasFiller(ParamFillerType::Constant, bias_const)
                    ->inputs({"fc6"})
                    ->outputs({"fc7"}))

            ->layer((new typename ReluLayer<Dtype>::Builder())
                    ->id(42)
                    ->name("relu7")
                    ->inputs({"fc7"})
                    ->outputs({"fc7"}))

            ->layer((new typename FullyConnectedLayer<Dtype>::Builder())
                    ->id(43)
                    ->name("fc8")
                    ->nOut(1000)
                    ->pDropout(0.0)
                    ->weightUpdateParam(1, 1)
                    ->biasUpdateParam(2, 0)
                    ->weightFiller(ParamFillerType::Gaussian, 0.1)
                    ->biasFiller(ParamFillerType::Constant, bias_const)
                    ->inputs({"fc7"})
                    ->outputs({"fc8"}))

            ->layer((new typename CrossEntropyWithLossLayer<Dtype>::Builder())
                    ->id(44)
                    ->name("celossKisti")
#if 1
                    ->withSigmoid(false)
#endif
                    ->inputs({"fc8", "label"})
                    ->outputs({"celossKisti"}))

#if 0
            ->layer((new typename SoftmaxWithLossLayer<Dtype>::Builder())
                    ->id(44)
                    ->name("loss")
                    ->inputs({"fc8", "label"})
                    ->outputs({"loss"}))
#endif

            ->build();

    return layersConfig;
#else
    return NULL;
#endif
}
#endif

template<typename Dtype>
void KISTIKeyword<Dtype>::run() {
    int networkID = PlanParser::loadNetwork(string(EXAMPLE_KISTIKEYWORD_NETWORK_FILEPATH));
    Network<Dtype>* network = Network<Dtype>::getNetworkFromID(networkID);
    network->build(100);
    network->run(false);

#if 0
    // loss layer of Discriminator GAN 
	const vector<string> lossList = { "celossKisti" };
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
				new NetworkMonitor("celossKisti", NetworkMonitor::PLOT_ONLY),
				})
			->lossLayers(lossList)
            ->optimizer(opt)
			->build();

	Util::printVramInfo();

 	Network<Dtype>* network = new Network<Dtype>(networkConfig);

    // (1) layer config를 만든다. 이 과정중에 layer들의 초기화가 진행된다.
	LayersConfig<Dtype>* layersConfig = createKistiVGG19NetLayersConfig();
 	network->setLayersConfig(layersConfig);

	// (2) network config 정보를 layer들에게 전달한다.
	for (uint32_t i = 0; i < layersConfig->_layers.size(); i++)
		layersConfig->_layers[i]->setNetworkConfig(networkConfig);

    // (3) 학습한다.
    for (int epoch = 0; epoch < 50; epoch++) {
        STDOUT_BLOCK(cout << "epoch #" << epoch << " starts" << endl;); 

        KistiInputLayer<Dtype>* etriInputLayer =
            dynamic_cast<KistiInputLayer<Dtype>*>(layersConfig->_firstLayers[0]);
        SASSERT0(etriInputLayer != NULL);

        CrossEntropyWithLossLayer<Dtype>* lossLayer =
            dynamic_cast<CrossEntropyWithLossLayer<Dtype>*>(layersConfig->_lastLayers[0]);
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

            const Dtype* inputData = etriInputLayer->_inputData[0]->host_data();
            const Dtype* outputData = lossLayer->_inputData[0]->host_data();
            const Dtype* outputLabel = lossLayer->_inputData[1]->host_data();
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

            const Dtype* inputData = etriInputLayer->_inputData[0]->host_data();
            const Dtype* outputData = lossLayer->_inputData[0]->host_data();
            const Dtype* outputLabel = lossLayer->_inputData[1]->host_data();
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
    DebugUtil<Dtype>::printNetworkEdges(stdout, "etri", layersConfig, 0);
    network->_backpropagation(0);
    DebugUtil<Dtype>::printNetworkEdges(stdout, "etri", layersConfig, 0);

    for (uint32_t i = 0; i < layersConfig->_learnableLayers.size(); i++) {
        layersConfig->_learnableLayers[i]->update();
    }
    DebugUtil<Dtype>::printNetworkEdges(stdout, "etri", layersConfig, 0);
#endif

#endif
}

template class KISTIKeyword<float>;
