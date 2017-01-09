#define SMOOTHL1LOSSLAYER_TEST 0

#if SMOOTHL1LOSSLAYER_TEST

#include "test_common.h"
#include "SmoothL1LossLayer.h"
#include "Cuda.h"

using namespace std;

SmoothL1LossLayer<float>* layer;
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
	cout.precision(10);
	cout.setf(ios::fixed);

	npzName = "/home/jkim/Documents/np_array/smoothl1losslayer.npz";
	dataNames = {"rpn_bbox_pred", "rpn_bbox_targets",
			"rpn_bbox_inside_weights", "rpn_bbox_outside_weights", "rpn_loss_bbox"};

	SmoothL1LossLayer<float>::Builder* builder = new typename SmoothL1LossLayer<float>::Builder();
	builder
		->id(1)
		->name("rpn_loss_bbox")
		->lossWeight(1.0f)
		->sigma(3.0f)
		->inputs({"rpn_bbox_pred", "rpn_bbox_targets", "rpn_bbox_inside_weights", "rpn_bbox_outside_weights"})
		->outputs({"rpn_loss_bbox"});

	layer = dynamic_cast<SmoothL1LossLayer<float>*>(builder->build());

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
	compare_data(layer->_outputData, dataMap, 0.000001f);
}

void backpropagation_test() {
	set_layer_data(layer->_inputs, dataMap, layer->_inputData);
	set_layer_data(layer->_outputs, dataMap, layer->_outputData);

	//print_datamap(dataMap);
	layer->backpropagation();
	//print_data(layer->_inputData);
	compare_grad(layer->_inputData, dataMap, 0.000001f);

	Data<float>::printConfig = true;
	layer->_inputData[0]->print_grad({}, false);
	Data<float>::printConfig = false;
}



#endif





