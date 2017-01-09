#define FRCNNTESTOUTPUTLAYER_TEST 0

#if FRCNNTESTOUTPUTLAYER_TEST

#include "test_common.h"
#include "FrcnnTestOutputLayer.h"

using namespace std;

FrcnnTestOutputLayer<float>* layer;
string npzName;
vector<string> dataNames;
map<string, Data<float>*> dataMap;

void setup();
void cleanup();

void feedforward_test();

int main() {
	setup();
	feedforward_test();
	cleanup();
}

void setup() {
	cout.precision(10);
	cout.setf(ios::fixed);

	npzName = "/home/jkim/Documents/np_array/frcnntestoutputlayer.npz";
	dataNames = {"rois", "im_info", "cls_prob", "bbox_pred"};

	FrcnnTestOutputLayer<float>::Builder* builder = new typename FrcnnTestOutputLayer<float>::Builder();
	builder
		->id(1)
		->name("test_output")
		//->maxPerImage(5)
		->thresh(0.001)
		->inputs({"rois", "im_info", "cls_prob", "bbox_pred"});
	layer = dynamic_cast<FrcnnTestOutputLayer<float>*>(builder->build());

	load_npz(npzName, dataNames, dataMap);
	//print_datamap(dataMap, 1);
}

void cleanup() {

}



void feedforward_test() {
	set_layer_data(layer->_inputs, dataMap, layer->_inputData);

	layer->feedforward();
	compare_data(layer->_outputData, dataMap);
}


#endif




