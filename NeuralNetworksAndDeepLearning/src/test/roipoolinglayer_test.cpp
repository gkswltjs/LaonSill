#define ROIPOOLINGLAYER_TEST 0

#if ROIPOOLINGLAYER_TEST

#include "test_common.h"
#include "RoIPoolingLayer.h"
#include "Cuda.h"

using namespace std;

RoIPoolingLayer<float>* layer;
string npzName;
vector<string> dataNames;
map<string, Data<float>*> dataMap;

void setup();
void cleanup();

void feedforward_test();
void backpropagation_test();


int main() {
	setup();
	feedforward_test();
	backpropagation_test();
	cleanup();
}

void setup() {
	npzName = "/home/jkim/Documents/np_array/roipoolinglayer.npz";
	dataNames = {"conv5", "rois", "pool5"};

	RoIPoolingLayer<float>::Builder* builder = new typename RoIPoolingLayer<float>::Builder();
	builder
		->id(1)
		->name("roi-data")
		->pooledW(6)
		->pooledH(6)
		->spatialScale(0.0625f)
		->inputs({"conv5", "rois"})
		->outputs({"pool5"});

	layer = dynamic_cast<RoIPoolingLayer<float>*>(builder->build());

	load_npz(npzName, dataNames, dataMap);

	//create_cuda_handle();
}

void cleanup() {
	//destroy_cuda_handle();
}



void feedforward_test() {
	set_layer_data(layer->_inputs, dataMap, layer->_inputData);
	set_layer_data(layer->_outputs, layer->_outputData);

	//print_datamap(dataMap);
	layer->feedforward();

	//print_data(layer->_outputData);
	compare_data(layer->_outputData, dataMap);
}

void backpropagation_test() {
	set_layer_data(layer->_inputs, dataMap, layer->_inputData);
	set_layer_data(layer->_outputs, dataMap, layer->_outputData);

	//print_datamap(dataMap);
	layer->backpropagation();
	//print_data(layer->_inputData);
	compare_grad(layer->_inputData, dataMap);
}


#endif




