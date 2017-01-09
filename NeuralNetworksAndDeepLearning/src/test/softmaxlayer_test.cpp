#define SOFTMAXLAYER_TEST 0

#if SOFTMAXLAYER_TEST

#include "test_common.h"
#include "SoftmaxLayer.h"
#include "Cuda.h"

using namespace std;

SoftmaxLayer<float>* layer;
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
	npzName = "/home/jkim/Documents/np_array/softmaxlayer.npz";
	dataNames = {"rpn_cls_score_reshape", "rpn_cls_prob"};
	load_npz(npzName, dataNames, dataMap);

	print_datamap(dataMap);

	SoftmaxLayer<float>::Builder* builder = new typename SoftmaxLayer<float>::Builder();
	builder
		->id(1)
		->name("softmax")
		->softmaxAxis(1)
		->inputs({"rpn_cls_score_reshape"})
		->outputs({"rpn_cls_prob"});
	layer = dynamic_cast<SoftmaxLayer<float>*>(builder->build());
	create_cuda_handle();
}

void cleanup() {
	destroy_cuda_handle();
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

	float error = 0.000001f;
	compare_data(layer->_inputData, dataMap, error);
	compare_grad(layer->_inputData, dataMap, error);
	compare_data(layer->_outputData, dataMap, error);
	compare_grad(layer->_outputData, dataMap, error);

	print_datamap(dataMap);
	print_data(layer->_outputData);
	print_data(layer->_inputData);
	layer->backpropagation();
	print_data(layer->_inputData);

	compare_grad(layer->_inputData, dataMap);


}

#endif




