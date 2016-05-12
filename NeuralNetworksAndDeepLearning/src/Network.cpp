/*
 * Network.cpp
 *
 *  Created on: 2016. 4. 20.
 *      Author: jhkim
 */

#include "Network.h"

#include <armadillo>
#include <iostream>
#include <vector>

#include "dataset/DataSample.h"
#include "dataset/DataSet.h"
#include "layer/HiddenLayer.h"
#include "layer/OutputLayer.h"
#include "Util.h"





Network::Network(int sizes[], int sizeCount, Layer **layers, DataSet *dataSet, NetworkListener *networkListener) {
	this->numLayers = sizeCount;
	this->sizes = sizes;

	//defaultWeightInitializer(weights, biases, true);

	this->layers = layers;
	this->dataSet = dataSet;
	//this->cost_fn = cost_fn;
	//this->activation_fn = activation_fn;
	this->networkListener = networkListener;
}

Network::~Network() {
	//deallocParameters(weights, biases);
}


/*
void Network::defaultWeightInitializer(vector<mat *> &weights, vector<vec *> &biases, bool init) {
	for(int i = 0; i < numLayers; i++) {
		// index를 맞추기 위해 0번에 dummy bias, weight 추가
		if(i == 0) {
			biases.push_back(new vec());
			weights.push_back((new mat()));
		} else {
			vec *pBias = new vec(sizes[i]);
			biases.push_back(pBias);

			mat *pWeight = new mat(sizes[i], sizes[i-1]);
			weights.push_back(pWeight);

			if(init) {
				pBias->randn();
				pWeight->randn();
				// initial point scaling
				(*pWeight) *= 1/sqrt(sizes[i-1]);
			} else {
				pBias->fill(0.0);
				pWeight->fill(0.0);
			}
			Util::printMat(pWeight, "weight");
			Util::printVec(pBias, "bias");
		}
	}
}


void Network::deallocParameters(vector<mat *> weights, vector<vec *> biases) {
	int size = weights.size();
	for(int i = 0; i < size; i++) {
		delete weights[i];
		delete biases[i];
	}
}
*/










void Network::sgd(int epochs, int miniBatchSize, double eta, double lambda) {
	int trainDataSize = dataSet->getTrainDataSize();
	int miniBatchesSize = trainDataSize / miniBatchSize;

	//vector<mat *> nabla_w;
	//vector<vec *> nabla_b;
	//defaultWeightInitializer(nabla_w, nabla_b, false);

	for(int i = 0; i < epochs; i++) {
		dataSet->shuffleTrainDataSet();

		for(int j = 0; j < miniBatchesSize; j++) {
			//for(int k = 1; k < numLayers; k++) {
			//	nabla_w[k]->fill(0.0);
			//	nabla_b[k]->fill(0.0);
			//}
			for(int k = 1; k < numLayers; k++) {
				(dynamic_cast<HiddenLayer *>(layers[k]))->reset_nabla();
			}
			updateMiniBatch(j, miniBatchSize, eta, lambda);
		}


		//dataSet->shuffleTestDataSet();
		if(dataSet->getTestDataSize() > 0) {
			cout << "Epoch " << i+1 << " " << evaluate() << " / " << dataSet->getTestDataSize() << endl;
		} else {
			cout << "Epoch " << i+1 << " complete." << endl;
		}


		/*
		cout << "Epoch " << i+1 << " training complete" << endl;
		if(networkListener) {
			//double validationCost = totalCost(dataSet->getValidationDataSet(), lambda);
			//double validationAccuracy = accuracy(dataSet->getValidationDataSet());
			//double trainCost = totalCost(dataSet->getTrainDataSet(), lambda);
			//double trainAccuracy = accuracy(dataSet->getTrainDataSet());
			double validationCost = 0;
			double validationAccuracy = accuracy(dataSet->getValidationDataSet());
			double trainCost = 0;
			double trainAccuracy = 0;
			networkListener->epochComplete(validationCost, validationAccuracy, trainCost, trainAccuracy);
		}
		*/
	}
	//deallocParameters(nabla_w, nabla_b);
}



void Network::updateMiniBatch(int nthMiniBatch, int miniBatchSize, double eta, double lambda) {

	int baseIndex = nthMiniBatch*miniBatchSize;
	for(int i = 0; i < miniBatchSize; i++) {
		backprop(dataSet->getTrainDataAt(baseIndex+i));
	}

	int n = dataSet->getTrainDataSize();
	for(int i = 1; i < numLayers; i++) {
		(dynamic_cast<HiddenLayer *>(layers[i]))->update(eta, lambda, n, miniBatchSize);
		// weight update에 L2 Regularization, Weight Decay 적용
		//(*weights[i]) = (1-eta*lambda/n)*(*weights[i]) - (eta/miniBatchSize)*(*nabla_w[i]);
		//(*weights[i]) -= eta/miniBatchSize*(*nabla_w[i]);
		//(*biases[i]) -= eta/miniBatchSize*(*nabla_b[i]);
	}
}



void Network::backprop(const DataSample &dataSample) {

	int lastLayerIndex = numLayers-1;

	// feedforward
	feedforward(dataSample.getData());


	// backward pass
	(dynamic_cast<OutputLayer *>(layers[lastLayerIndex]))->cost(dataSample.getTarget(), layers[lastLayerIndex-1]->getActivation());

	for(int i = lastLayerIndex-1; i > 0; i--) {
		(dynamic_cast<HiddenLayer *>(layers[i]))->backpropagation((dynamic_cast<HiddenLayer *>(layers[i+1]))->getWeight(),
				(dynamic_cast<HiddenLayer *>(layers[i+1]))->getDelta(), layers[i-1]->getActivation());
	}


	/*
	vector<const vec *> activations;
	activations.push_back(activation);

	vector<const vec *> zs;
	// index를 맞추기 위해 dummy z를 0번에 추가
	zs.push_back(new vec());

	// z[i] = weight[i]*activation[i-1]+b[i]
	// activation[i] = sigmoid(z[i])
	for(int i = 1; i < numLayers; i++) {
		vec *b = biases[i];
		mat *w = weights[i];

		Util::printVec(activation, "activation");
		Util::printVec(b, "bias");
		Util::printMat(w, "weight");

		vec *z = new vec((*w)*(*activation) + (*b));
		Util::printVec(z, "z");
		zs.push_back(z);

		activation = new vec(sigmoid(z));
		Util::printVec(activation, "activation");
		activations.push_back(activation);
	}

	// backward pass
	int lastLayerIndex = numLayers-1;
	// δL = (aL−y) ⊙ σ′(zL)
	//vec delta = costDerivative(activations[lastLayerIndex], dataSample->getTarget()) % sigmoidPrime(activations[lastLayerIndex]);
	vec delta = cost->delta(zs[lastLayerIndex], activations[lastLayerIndex], dataSample->getTarget());
	Util::printVec(&delta, "delta");

	// ∂C / ∂b = δ
	// ∂C / ∂w = ain * δout
	Util::printVec(nabla_b[lastLayerIndex], "bias");
	Util::printMat(nabla_w[lastLayerIndex], "weight");
	Util::printVec(activations[lastLayerIndex-1], "activation");
	(*nabla_b[lastLayerIndex]) += delta;
	(*nabla_w[lastLayerIndex]) += delta*activations[lastLayerIndex-1]->t();
	Util::printVec(nabla_b[lastLayerIndex], "bias");
	Util::printMat(nabla_w[lastLayerIndex], "weight");

	for(int l = lastLayerIndex-1; l >= 1; l--) {
		// δl = ((wl+1)T * δl+1) ⊙ σ′(zl)
		vec sp = sigmoidPrime(activations[l]);
		delta = weights[l+1]->t()*delta % sp;
		Util::printVec(&delta, "delta");
		(*nabla_b[l]) += delta;
		(*nabla_w[l]) += delta*activations[l-1]->t();
		Util::printVec(nabla_b[l], "bias");
		Util::printMat(nabla_w[l], "weight");
	}

	// dealloc
	for(int i = 0; i < numLayers-1; i++) {
		delete activations[i+1];
		delete zs[i];
	}
	*/
}

/*
vec Network::costDerivative(const vec *outputActivation, const vec *y) {
	Util::printVec(outputActivation, "outputActivation");
	Util::printVec(y, "y");

	vec costDerivative = (*outputActivation) - (*y);
	Util::printVec(&costDerivative, "costDerivative");

	return costDerivative;
}


vec Network::sigmoid(const vec *z) {
	vec temp = ones<vec>(z->n_rows);
	return temp / (1.0+exp(-1*(*z)));
}

// activation이 이미 계산된 상태이므로 z가 아닌 activation을 통해 simoid prime을 계산
vec Network::sigmoidPrime(const vec *activation) {
	Util::printVec(activation, "activation");
	vec result = (*activation)%(1.0-*activation);
	Util::printVec(&result, "result");
	return result;
}
*/






void Network::feedforward(const vec &input) {
	/*
	Util::printVec(x, "x");
	vec activation(*x);
	Util::printVec(&activation, "activation");

	for(int i = 1; i < numLayers; i++) {
		//Util::printMat(weights[i], "weight");
		//Util::printVec(biases[i], "bias");

		vec z = (*weights[i])*activation+(*biases[i]);
		Util::printVec(&z, "z");

		activation = sigmoid(&z);
		Util::printVec(&activation, "activation");
	}
	return activation;
	*/

	layers[0]->feedforward(input);
	for(int i = 1; i < numLayers; i++) {
		layers[i]->feedforward(layers[i-1]->getActivation());
	}

}






/*
double Network::totalCost(const vector<const DataSample *> &dataSet, double lambda) {
	double cost = 0.0;
	int dataSize = dataSet.size();

	for(int i = 0; i < dataSize; i++) {
		vec activation = feedforward(dataSet[i]->getData());
		cost += this->cost->fn(&activation, dataSet[i]->getTarget());
	}
	cost /= dataSize;

	// add weight decay term of cost
	for(int i = 1; i < numLayers; i++) {
		cost += 0.5*(lambda/dataSize)*accu(square(*weights[i]));
	}
	return cost;
}



double Network::accuracy(const vector<const DataSample *> &dataSet) {
	int total = 0;
	int dataSize = dataSet.size();
	for(int i = 0; i < dataSize; i++) {
		const DataSample *dataSample = dataSet[i];
		Util::printVec(dataSample->getData(), "data");
		Util::printVec(dataSample->getTarget(), "target");
		total += testEvaluateResult(feedforward(dataSample->getData()), dataSample->getTarget());
	}
	return total/(double)dataSize;
}
*/




int Network::testEvaluateResult(const vec &output, const vec &y) {
	//Util::printVec(&evaluateResult, "result");
	//Util::printVec(y, "y");

	uword rrow, yrow;
	output.max(rrow);
	y.max(yrow);

	if(rrow == yrow) return 1;
	else return 0;
}



/*
void Network::save(string filename) {

}

void Network::load(string filename) {
}
*/



int Network::evaluate() {
	int testResult = 0;
	//bool printBak = Util::getPrint();
	//Util::setPrint(true);
	int testDataSize = dataSet->getTestDataSize();
	for(int i = 0; i < testDataSize; i++) {
		//const DataSample *testData = dataSet->getTestDataAt(i);
		const DataSample &testData = dataSet->getTestDataAt(i);
		//Util::printVec(testData->getData(), "data");
		//Util::printVec(testData->getTarget(), "target");

		feedforward(testData.getData());
		testResult += testEvaluateResult(layers[numLayers-1]->getActivation(), testData.getTarget());
	}
	//Util::setPrint(printBak);

	return testResult;
}
















































