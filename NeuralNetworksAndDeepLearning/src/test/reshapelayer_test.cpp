#define RESHAPELAYER_TEST 0

#if RESHAPELAYER_TEST

#include "test_common.h"
#include "ReshapeLayer.h"

using namespace std;

ReshapeLayer<float>* layer;
string npzName;
vector<string> dataNames;
map<string, Data<float>*> dataMap;

void setup();
void feedforward_test();
void backpropagation_test();

int main() {
	setup();
	//feedforward_test();
	backpropagation_test();
}

void setup() {
	npzName = "/home/jkim/Documents/np_array/reshapelayer.npz";
	dataNames = {"rpn_cls_score", "rpn_cls_score_reshape"};

	ReshapeLayer<float>::Builder* builder = new typename ReshapeLayer<float>::Builder();
	builder
		->id(1)
		->name("rpn_cls_score_reshape")
		->shape({0, 2, -1, 0})
		->inputs({"rpn_cls_score"})
		->outputs({"rpn_cls_score_reshape"});

	layer = dynamic_cast<ReshapeLayer<float>*>(builder->build());

	load_npz(npzName, dataNames, dataMap);
}



void feedforward_test() {
	set_layer_data(layer->_inputs, dataMap, layer->_inputData);
	set_layer_data(layer->_outputs, layer->_outputData);

	print_datamap(dataMap);

	layer->feedforward();

	print_data(layer->_outputData);

	compare_data(layer->_outputData, dataMap, float(0.0001));


	//Data<float>::printConfig = true;
	//for (uint32_t i = 0; i < layer->_outputData.size(); i++)
	//	layer->_outputData[i]->print_data({}, false);
	//Data<float>::printConfig = false;
}


void backpropagation_test() {
	set_layer_data(layer->_inputs, dataMap, layer->_inputData);
	set_layer_data(layer->_outputs, dataMap, layer->_outputData);

	layer->backpropagation();
	compare_grad(layer->_inputData, dataMap);
}



#endif




