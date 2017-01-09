#define SOFTMAXWITHLOSSLAYER_TEST 0

#if SOFTMAXWITHLOSSLAYER_TEST

#include "test_common.h"
#include "SoftmaxWithLossLayer.h"
#include "Cuda.h"

using namespace std;

SoftmaxWithLossLayer<float>* layer;
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
	npzName = "/home/jkim/Documents/np_array/softmaxwithlosslayer.npz";
	dataNames = {"rpn_cls_score_reshape", "rpn_labels", "rpn_cls_loss"};

	SoftmaxWithLossLayer<float>::Builder* builder = new typename SoftmaxWithLossLayer<float>::Builder();
	builder
		->id(1)
		->name("rpn_loss_cls")
		->propDown({true, false})
		->lossWeight(1.0f)
		->ignoreLabel(-1)
		->normalize(true)
		->softmaxAxis(1)
		->inputs({"rpn_cls_score_reshape", "rpn_labels"})
		->outputs({"rpn_cls_loss"});

	layer = dynamic_cast<SoftmaxWithLossLayer<float>*>(builder->build());

	load_npz(npzName, dataNames, dataMap);

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
	compare_data(layer->_outputData, dataMap, float(0.0001));
}


void backpropagation_test() {
	set_layer_data(layer->_inputs, dataMap, layer->_inputData);
	set_layer_data(layer->_outputs, dataMap, layer->_outputData);

	print_datamap(dataMap);
	layer->backpropagation();
	print_data(layer->_inputData);
	compare_grad(layer->_inputData, dataMap, float(0.0001));

}

#endif




