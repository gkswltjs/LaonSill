#define ROITESTINPUTLAYER_TEST 0

#if ROITESTINPUTLAYER_TEST

#include "common.h"
#include "test_common.h"
#include "RoITestInputLayer.h"

using namespace std;

RoITestInputLayer<float>* layer;
string npzName;
vector<string> dataNames;
map<string, Data<float>*> dataMap;

void setup();
void feedforward_test();
void cleanup();

int main() {
	setup();
	feedforward_test();
	cleanup();
}


void setup() {
	npzName = "/home/jkim/Documents/np_array/roiinputlayer.npz";
	dataNames = {"data", "im_info", "gt_boxes"};

	// layer
	RoITestInputLayer<float>::Builder* builder = new typename RoITestInputLayer<float>::Builder();
	builder
		->id(0)
		->name("input-data")
		->numClasses(21)
		->pixelMeans({0.4815f, 0.4547f, 0.4038f})		// RGB
		->outputs({"data", "im_info"});
	layer = dynamic_cast<RoITestInputLayer<float>*>(builder->build());
	load_npz(npzName, dataNames, dataMap);
}

void feedforward_test() {
	set_layer_data(layer->_outputs, layer->_outputData);

	layer->feedforward();
	compare_data(layer->_outputData, dataMap, float(0.01));
}

void cleanup() {

}


#endif




