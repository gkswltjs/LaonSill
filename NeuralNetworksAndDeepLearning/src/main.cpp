#include <iostream>
#include <armadillo>

#include "network/Network.h"
#include "network/GoogLeNet.h"
#include "network/NeuralNetSingle.h"
#include "network/NeuralNetDouble.h"
#include "network/GoogLeNetMnist.h"
#include "network/ConvNetSingle.h"
#include "network/ConvNetDouble.h"
#include "network/InceptionNetSingle.h"
#include "dataset/MnistDataSet.h"
#include "dataset/MockDataSet.h"
#include "dataset/Cifar10DataSet.h"
#include "Util.h"
#include "pooling/Pooling.h"
#include "pooling/MaxPooling.h"
#include "pooling/AvgPooling.h"
#include "cost/CrossEntropyCost.h"
#include "cost/LogLikelihoodCost.h"
#include "monitor/NetworkMonitor.h"
#include "exception/Exception.h"

#include "layer/Layer.h"
#include "layer/InputLayer.h"
#include "layer/FullyConnectedLayer.h"
#include "layer/ConvLayer.h"
#include "layer/PoolingLayer.h"
#include "layer/SigmoidLayer.h"
#include "layer/SoftmaxLayer.h"
#include "layer/LRNLayer.h"
#include "layer/InceptionLayer.h"
#include "layer/DepthConcatLayer.h"
#include "activation/Activation.h"
#include "activation/Sigmoid.h"
#include "activation/ReLU.h"

#include "Timer.h"
#include "cuda/Cuda.h"

using namespace std;
using namespace arma;

// Armadillo documentation is available at:
// http://arma.sourceforge.net/docs.html


void network_test();
void cuda_test();

int main(int argc, char** argv) {
	cout << "main" << endl;
	cout.precision(11);
	cout.setf(ios::fixed);

	//network_test();
	cuda_test();

	cout << "end" << endl;
	return 0;
}


void cuda_test() {
	Cuda::create(0);

	MockDataSet *dataSet = new MockDataSet();
	dataSet->load();

	const DataSample &dataSample = dataSet->getTestDataAt(0);
	InputLayer *inputLayer = new InputLayer("input", io_dim(10, 10, 3));
	inputLayer->feedforward(0, dataSample.getData());

	//inputLayer->feedforward(0, )

	Cuda::destroy();
}





void network_test() {
	arma_rng::set_seed_random();

	bool debug = false;
	double validationSetRatio = 1.0/6.0;

	if(!debug) {
		Util::setPrint(false);

		DataSet *mnistDataSet = new MnistDataSet(validationSetRatio);
		mnistDataSet->load();

		//Network *network = new ConvNetSingle();
		//network->setDataSet(mnistDataSet);
		//network->sgd(10, 10);
	} else {
		Util::setPrint(true);

		MockDataSet *dataSet = new MockDataSet();
		dataSet->load();
	}

}






