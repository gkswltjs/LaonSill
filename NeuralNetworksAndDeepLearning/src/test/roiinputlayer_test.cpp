#define ROIINPUTLAYER_TEST 0

#if ROIINPUTLAYER_TEST

#include "common.h"
#include "test_common.h"
#include "RoIInputLayer.h"

using namespace std;

RoIInputLayer<float>* layer;
string npzName;
vector<string> dataNames;
map<string, Data<float>*> dataMap;

void setup();
void feedforward_test();


int main() {
	setup();
	feedforward_test();
}


void setup() {

	npzName = "/home/jkim/Documents/np_array/roiinputlayer.npz";
	dataNames = {"data", "im_info", "gt_boxes"};

	// layer
	RoIInputLayer<float>::Builder* builder = new typename RoIInputLayer<float>::Builder();
	builder
		->id(0)
		->name("input-data")
		->numClasses(21)
#if TEST_MODE
		->pixelMeans({102.9801f, 115.9465f, 122.7717f})		// BGR
#else
		->pixelMeans({0.4815f, 0.4547f, 0.4038f})		// RGB
#endif
		->outputs(dataNames);
	layer = dynamic_cast<RoIInputLayer<float>*>(builder->build());

	load_npz(npzName, dataNames, dataMap);

}



void feedforward_test() {
	set_layer_data(layer->_outputs, layer->_outputData);

	layer->feedforward();
	compare_data(layer->_outputData, dataMap, float(0.01));
	//Data<float>::printConfig = true;
	//for (uint32_t i = 0; i < layer->_outputData.size(); i++)
	//	layer->_outputData[i]->print_data({}, false);
	//Data<float>::printConfig = false;

}



#endif




