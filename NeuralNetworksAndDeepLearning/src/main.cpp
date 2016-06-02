#include <iostream>
#include <armadillo>

#include "network/Network.h"
#include "network/GoogLeNet.h"
#include "network/NeuralNetSingle.h"
#include "network/GoogLeNetMnist.h"
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

using namespace std;
using namespace arma;

// Armadillo documentation is available at:
// http://arma.sourceforge.net/docs.html


void armadillo_test();
void network_test();


int main(int argc, char** argv) {
	cout << "main" << endl;

	cout.precision(11);
	cout.setf(ios::fixed);

	//network_test();


	/*
	rcube c = randn<rcube>(4, 6, 2);
	c.print("c:");

	for(int k = 0; k < c.n_slices; k++) {
	for(int i = 0; i < c.n_rows; i++) {
		for(int j = 0; j < c.n_cols; j++) {
			cout << C_MEM(c, i, j, k) << ", ";
		}
		cout << endl;
	}
	}
	cout << endl;
	for(int k = 0; k < c.n_slices; k++) {
	for(int i = 0; i < c.n_rows; i++) {
		for(int j = 0; j < c.n_cols; j++) {
			cout << C_MEMPTR(c, i, j, k) << ", ";
		}
		cout << endl;
	}
	}
	cout << endl;

	double temp;
	for(int k = 0; k < c.n_slices; k++) {
	for(int i = 0; i < c.n_rows; i++) {
		for(int j = 0; j < c.n_cols; j++) {
			temp = C_MEM(c, i, j, k) * 10;
			C_MEMPTR(c, i, j, k) = temp;
			//cout << C_MEMPTR(c, i, j, 0) << ", ";
		}
		//cout << endl;
	}
	}

	c.print("c:");

	cout << endl;

	//rcube input = randn<rcube>(14, 14, 2);
	//input.memptr()[0]
	 */






	rcube input = randn<rcube>(7, 7, 1);
	ucube pool_map = zeros<ucube>(size(input));
	rcube output = zeros<rcube>(1, 1, 1);



	Util::printCube(input, "input:");

	AvgPooling ap;
	ap.pool(pool_dim(7, 7, 4), input, pool_map, output);

	Util::printCube(output, "output:");

	ap.d_pool(pool_dim(7, 7, 4), output, pool_map, input);

	Util::printCube(input, "input:");

	return 0;
}




void network_test() {
	arma_rng::set_seed_random();

	bool debug = true;
	double validationSetRatio = 1.0/6.0;

	if(!debug) {
		Util::setPrint(false);

		// DataSet은 memory를 크게 차지할 수 있으므로 heap에 생성
		DataSet *mnistDataSet = new MnistDataSet(validationSetRatio);
		mnistDataSet->load();
		//DataSet *cifar10DataSet = new Cifar10DataSet();
		//cifar10DataSet->load();

		Network *network = new NeuralNetSingle();
		network->setDataSet(mnistDataSet);
		network->sgd(10, 10, 0.1, 5.0);

		//Network *googlenet = new GoogLeNetMnist(0);
		//googlenet->setDataSet(mnistDataSet);
		//googlenet->sgd(10, 10, 0.01, 5.0);

	} else {
		Util::setPrint(true);

		//Activation *relu1 = new ReLU(io_dim(8, 8, 5));
		//Activation *relu2 = new ReLU(io_dim(10, 1, 1));

		MockDataSet *dataSet = new MockDataSet();
		dataSet->load();

		/*
		Network *googlenet = new GoogLeNet(0);
		googlenet->setDataSet(dataSet);
		googlenet->sgd(1, 10, 0.1, lambda);
		//googlenet.sgd(5, 2, 3.0, lambda);
		 */


		/*
		InputLayer *inputLayer = new InputLayer("input", io_dim(10, 10, 3));
		HiddenLayer *inceptionLayer = new InceptionLayer("inception", io_dim(10, 10, 3), io_dim(10, 10, 12), 3, 2, 3, 2, 3, 3);
		HiddenLayer *pool1Layer = new PoolingLayer("pool1", io_dim(10, 10, 12), pool_dim(3, 3, 2), maxPooling);
		HiddenLayer *fc1Layer = new FullyConnectedLayer("fc1", 5*5*12, 100, 0.5, sigmoid);
		OutputLayer *softmaxLayer = new SoftmaxLayer("softmax", 100, 10, 0.5);

		Network::addLayerRelation(inputLayer, inceptionLayer);
		Network::addLayerRelation(inceptionLayer, pool1Layer);
		Network::addLayerRelation(pool1Layer, fc1Layer);
		Network::addLayerRelation(fc1Layer, softmaxLayer);
		*/


		/*
		InputLayer *inputLayer = new InputLayer("input", io_dim(10, 10, 3));
		HiddenLayer *inceptionLayer = new InceptionLayer("inception", io_dim(10, 10, 3), io_dim(10, 10, 12), 3, 2, 3, 2, 3, 3);
		OutputLayer *softmaxLayer = new SoftmaxLayer("softmax", 10*10*12, 10, 0.5);

		Network::addLayerRelation(inputLayer, inceptionLayer);
		Network::addLayerRelation(inceptionLayer, softmaxLayer);
		*/




		/*
		InputLayer *inputLayer = new InputLayer("input", io_dim(10, 10, 1));
		HiddenLayer *conv1Layer = new ConvLayer("conv1", io_dim(10, 10, 1), filter_dim(3, 3, 1, 2, 1), sigmoid);
		HiddenLayer *pool1Layer = new PoolingLayer("pool1", io_dim(10, 10, 2), pool_dim(3, 3, 2), maxPooling);
		HiddenLayer *conv2Layer = new ConvLayer("conv2", io_dim(5, 5, 2), filter_dim(3, 3, 2, 4, 1), sigmoid);
		HiddenLayer *pool2Layer = new PoolingLayer("pool2", io_dim(5, 5, 4), pool_dim(3, 3, 2), maxPooling);
		HiddenLayer *fc1Layer = new FullyConnectedLayer("fc1", 2*2*4, 10, 0.5, sigmoid);
		OutputLayer *softmaxLayer = new SoftmaxLayer("softmax", 10, 10, 0.5);

		Network::addLayerRelation(inputLayer, conv1Layer);
		Network::addLayerRelation(conv1Layer, pool1Layer);
		Network::addLayerRelation(pool1Layer, conv2Layer);
		Network::addLayerRelation(conv2Layer, pool2Layer);
		Network::addLayerRelation(pool2Layer, fc1Layer);
		Network::addLayerRelation(fc1Layer, softmaxLayer);
		*/

		/*
		InputLayer *inputLayer = new InputLayer("input", io_dim(10, 10, 1));
		HiddenLayer *fc1Layer = new FullyConnectedLayer("fc1", 100, 10, 0.5, sigmoid);
		HiddenLayer *fc21Layer = new FullyConnectedLayer("fc21", 10, 10, 0.5, sigmoid);
		HiddenLayer *fc22Layer = new FullyConnectedLayer("fc22", 10, 10, 0.5, sigmoid);
		HiddenLayer *fc23Layer = new FullyConnectedLayer("fc23", 10, 10, 0.5, sigmoid);
		HiddenLayer *fc3Layer = new FullyConnectedLayer("fc3", 10, 10, 0.5, sigmoid);
		OutputLayer *softmax1Layer = new SoftmaxLayer("softmax1", 10, 10, 0.5);
		OutputLayer *softmax2Layer = new SoftmaxLayer("softmax2", 10, 10, 0.5);

		Network::addLayerRelation(inputLayer, fc1Layer);
		Network::addLayerRelation(fc1Layer, fc21Layer);
		Network::addLayerRelation(fc1Layer, fc22Layer);
		Network::addLayerRelation(fc1Layer, fc23Layer);
		Network::addLayerRelation(fc21Layer, fc3Layer);
		Network::addLayerRelation(fc22Layer, fc3Layer);
		Network::addLayerRelation(fc23Layer, fc3Layer);
		Network::addLayerRelation(fc3Layer, softmax1Layer);
		Network::addLayerRelation(fc1Layer, softmax2Layer);
		*/

		InputLayer *inputLayer = new InputLayer("input", io_dim(10, 10, 1));
		HiddenLayer *fc1Layer = new FullyConnectedLayer("fc1", 10*10*1, 100, 0.5, new ReLU(io_dim(100, 1, 1)));
		OutputLayer *softmaxLayer = new SoftmaxLayer("softmax", 100, 10, 0.5);

		Network::addLayerRelation(inputLayer, fc1Layer);
		Network::addLayerRelation(fc1Layer, softmaxLayer);

		Network network(inputLayer, dataSet, 0);
		network.addOutputLayer(softmaxLayer);
		//network.addOutputLayer(softmax2Layer);

		network.sgd(5, 2, 3.0, 5.0);
	}

}









void armadillo_test()
{
	cout << "Armadillo version: " << arma_version::as_string() << endl;

	mat A(2, 3); // directly specify the matrix size (elements are uninitialised)

	cout << "A.n_rows: " << A.n_rows << endl; // .n_rows and .n_cols are read only
	cout << "A.n_cols: " << A.n_cols << endl;

	A(1, 2) = 456.0;  // directly access an element (indexing starts at 0)
	A.print("A:");

	A = 5.0;         // scalars are treated as a 1x1 matrix
	A.print("A:");

	A.set_size(4, 5); // change the size (data is not preserved)

	A.fill(5.0);     // set all elements to a particular value
	A.print("A:");

	// endr indicates "end of row"
	A << 0.165300 << 0.454037 << 0.995795 << 0.124098 << 0.047084 << endr
			<< 0.688782 << 0.036549 << 0.552848 << 0.937664 << 0.866401 << endr
			<< 0.348740 << 0.479388 << 0.506228 << 0.145673 << 0.491547 << endr
			<< 0.148678 << 0.682258 << 0.571154 << 0.874724 << 0.444632 << endr
			<< 0.245726 << 0.595218 << 0.409327 << 0.367827 << 0.385736 << endr;

	A.print("A:");

	// determinant
	// cout << "det(A): " << det(A) << endl;

	// inverse
	// cout << "inv(A): " << endl << inv(A) << endl;

	// save matrix as a text file
	A.save("A.txt", raw_ascii);

	// load from file
	mat B;
	B.load("A.txt");

	// submatrices
	cout << "B( span(0,2), span(3,4) ):" << endl << B(span(0, 2), span(3, 4))
			<< endl;

	cout << "B( 0,3, size(3,2) ):" << endl << B(0, 3, size(3, 2)) << endl;

	cout << "B.row(0): " << endl << B.row(0) << endl;

	cout << "B.col(1): " << endl << B.col(1) << endl;

	// transpose
	cout << "B.t(): " << endl << B.t() << endl;

	// maximum from each column (traverse along rows)
	cout << "max(B): " << endl << max(B) << endl;

	// maximum from each row (traverse along columns)
	cout << "max(B,1): " << endl << max(B, 1) << endl;

	// maximum value in B
	cout << "max(max(B)) = " << max(max(B)) << endl;

	// sum of each column (traverse along rows)
	cout << "sum(B): " << endl << sum(B) << endl;

	// sum of each row (traverse along columns)
	cout << "sum(B,1) =" << endl << sum(B, 1) << endl;

	// sum of all elements
	cout << "accu(B): " << accu(B) << endl;

	// trace = sum along diagonal
	cout << "trace(B): " << trace(B) << endl;

	// generate the identity matrix
	mat C = eye < mat > (4, 4);

	// random matrix with values uniformly distributed in the [0,1] interval
	mat D = randu < mat > (4, 4);
	D.print("D:");

	// row vectors are treated like a matrix with one row
	rowvec r;
	r << 0.59119 << 0.77321 << 0.60275 << 0.35887 << 0.51683;
	r.print("r:");

	// column vectors are treated like a matrix with one column
	vec q;
	q << 0.14333 << 0.59478 << 0.14481 << 0.58558 << 0.60809;
	q.print("q:");

	// convert matrix to vector; data in matrices is stored column-by-column
	vec v = vectorise(A);
	v.print("v:");

	// dot or inner product
	cout << "as_scalar(r*q): " << as_scalar(r * q) << endl;

	// outer product
	cout << "q*r: " << endl << q * r << endl;

	// multiply-and-accumulate operation (no temporary matrices are created)
	cout << "accu(A % B) = " << accu(A % B) << endl;

	// example of a compound operation
	B += 2.0 * A.t();
	B.print("B:");

	// imat specifies an integer matrix
	imat AA;
	imat BB;

	AA << 1 << 2 << 3 << endr << 4 << 5 << 6 << endr << 7 << 8 << 9;
	BB << 3 << 2 << 1 << endr << 6 << 5 << 4 << endr << 9 << 8 << 7;

	// comparison of matrices (element-wise); output of a relational operator is a umat
	umat ZZ = (AA >= BB);
	ZZ.print("ZZ:");

	// cubes ("3D matrices")
	cube Q(B.n_rows, B.n_cols, 2);

	Q.slice(0) = B;
	Q.slice(1) = 2.0 * B;

	Q.print("Q:");

	// 2D field of matrices; 3D fields are also supported
	field < mat > F(4, 3);

	for (uword col = 0; col < F.n_cols; ++col)
		for (uword row = 0; row < F.n_rows; ++row) {
			F(row, col) = randu < mat > (2, 3); // each element in field<mat> is a matrix
		}

	F.print("F:");

}

