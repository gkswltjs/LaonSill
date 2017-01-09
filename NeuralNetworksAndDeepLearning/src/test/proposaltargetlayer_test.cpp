#define PROPOSALTARGETLAYER_TEST 0

#if PROPOSALTARGETLAYER_TEST

//#include "nms/gpu_nms.hpp"

#include "test_common.h"
#include "ProposalTargetLayer.h"
#include "Cuda.h"

using namespace std;

ProposalTargetLayer<float>* layer;
string npzName;
vector<string> dataNames;
map<string, Data<float>*> dataMap;

void setup();
void cleanup();

void feedforward_test();
void backpropagation_test();

void _restoreShape();

int main() {
	setup();
	feedforward_test();
	//backpropagation_test();
	cleanup();
}

void setup() {
	npzName = "/home/jkim/Documents/np_array/proposaltargetlayer.npz";
	dataNames = {"rpn_rois", "gt_boxes", "rois", "labels", "bbox_targets",
			"bbox_inside_weights", "bbox_outside_weights"};

	ProposalTargetLayer<float>::Builder* builder = new typename ProposalTargetLayer<float>::Builder();
	builder
		->id(1)
		->name("roi-data")
		->numClasses(21)
		->inputs({"rpn_rois", "gt_boxes"})
		->outputs({"rois", "labels", "bbox_targets", "bbox_inside_weights", "bbox_outside_weights"});

	layer = dynamic_cast<ProposalTargetLayer<float>*>(builder->build());

	load_npz(npzName, dataNames, dataMap);

	//create_cuda_handle();
}

void cleanup() {
	//destroy_cuda_handle();
}



void feedforward_test() {
	set_layer_data(layer->_inputs, dataMap, layer->_inputData);
	set_layer_data(layer->_outputs, layer->_outputData);

	print_datamap(dataMap);
	layer->feedforward();
	_restoreShape();

	print_data(layer->_outputData);
	compare_data(layer->_outputData, dataMap);
}

void backpropagation_test() {
	set_layer_data(layer->_inputs, dataMap, layer->_inputData);
	set_layer_data(layer->_outputs, dataMap, layer->_outputData);

	//print_datamap(dataMap);
	layer->backpropagation();
	//print_data(layer->_outputData);

	_restoreShape();

	compare_grad(layer->_inputData, dataMap);
}

void _restoreShape() {

	for (uint32_t i = 2; i < 5; i++) {
		const uint32_t numTargets = layer->_outputData[i]->getShape(0);
		const uint32_t numTargetElems = layer->_outputData[i]->getShape(2);
		layer->_outputData[i]->reshape({1, 1, numTargets, numTargetElems});
	}
}


#endif




