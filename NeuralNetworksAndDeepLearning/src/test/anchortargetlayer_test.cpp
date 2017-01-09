#define ANCHORTARGETLAYER_TEST 0

#if ANCHORTARGETLAYER_TEST

#include "test_common.h"
#include "AnchorTargetLayer.h"

using namespace std;

AnchorTargetLayer<float>* layer;
string npzName;
vector<string> dataNames;
map<string, Data<float>*> dataMap;

void setup();
void bboxTarget_test();
void feedforward_test();
void backpropagation_test();

int main() {
	setup();
	//feedforward_test();
	backpropagation_test();
	//bboxTarget_test();
}

void setup() {
	cout.precision(10);
	cout.setf(ios::fixed);

	npzName = "/home/jkim/Documents/np_array/anchortargetlayer.npz";
	dataNames = {"rpn_cls_score", "gt_boxes", "im_info", "data",
        "rpn_labels", "rpn_bbox_targets", "rpn_bbox_inside_weights",
        "rpn_bbox_outside_weights"};

	AnchorTargetLayer<float>::Builder* builder = 
        new typename AnchorTargetLayer<float>::Builder();
	builder
		->id(12)
		->name("rpn-data")
		->featStride(16)
		->inputs({"rpn_cls_score", "gt_boxes", "im_info", "data"})
		->outputs({"rpn_labels", "rpn_bbox_targets", "rpn_bbox_inside_weights",
                   "rpn_bbox_outside_weights"});

	layer = dynamic_cast<AnchorTargetLayer<float>*>(builder->build());

	load_npz(npzName, dataNames, dataMap);
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

	Data<float>::printConfig = true;

	Data<float>::printConfig = false;

	compare_grad(layer->_inputData, dataMap);
}

void bboxTarget_test() {
	Data<float>* bboxTarget = new Data<float>("bboxTargets");
	bboxTarget->load("/home/jkim/Documents/bboxTargets.data");
	bboxTarget->_name = "bboxTargets";
	Data<float>::printConfig = true;
	//bboxTarget->print_data({}, false);
	Data<float>::printConfig = false;
	vector<Data<float>*> testDataList;
	testDataList.push_back(bboxTarget);

	npzName = "/home/jkim/Documents/bboxTargets.npz";
	dataNames = {"bboxTargets"};
	load_npz(npzName, dataNames, dataMap);
	//print_datamap(dataMap);



	compare_data(testDataList, dataMap);

}



#endif




