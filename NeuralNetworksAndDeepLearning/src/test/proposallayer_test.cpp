#define PROPOSALLAYER_TEST 0

#if PROPOSALLAYER_TEST

//#include "nms/gpu_nms.hpp"

#include "test_common.h"
#include "ProposalLayer.h"
#include "Cuda.h"
#include "frcnn_common.h"

using namespace std;

ProposalLayer<float>* layer;
string npzName;
vector<string> dataNames;
map<string, Data<float>*> dataMap;

void setup();
void cleanup();

void feedforward_test();
void backpropagation_test();

void nms_test();



int main() {
	//setup();
	//feedforward_test();
	//backpropagation_test();
	//cleanup();

	nms_test();
}

void setup() {
	cout.precision(20);
	cout.setf(ios::fixed);

	npzName = "/home/jkim/Documents/np_array/proposallayer.npz";
	dataNames = {"rpn_cls_prob_reshape", "rpn_bbox_pred", "im_info", "rpn_rois"};

	ProposalLayer<float>::Builder* builder = new typename ProposalLayer<float>::Builder();
	builder
		->id(1)
		->name("proposal")
		->featStride(16)
		->inputs({"rpn_cls_prob_reshape", "rpn_bbox_pred", "im_info"})
		->outputs({"rpn_rois"});

	layer = dynamic_cast<ProposalLayer<float>*>(builder->build());

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
	//print_data(layer->_outputData);
	compare_data(layer->_outputData, dataMap, 0.0001f);
}

void backpropagation_test() {
	set_layer_data(layer->_inputs, dataMap, layer->_inputData);
	set_layer_data(layer->_outputs, dataMap, layer->_outputData);

	print_datamap(dataMap);
	layer->backpropagation();
	print_data(layer->_inputData);
	compare_grad(layer->_inputData, dataMap, 0.0001f);
}


void nms_test() {
	npzName = "/home/jkim/Documents/np_array/save/nms.npz";
	dataNames = {"arr_0"};
	load_npz(npzName, dataNames, dataMap);

	print_datamap(dataMap);

	typename std::map<std::string, Data<float>*>::iterator it;
	it = dataMap.find(dataNames[0]);
	assert(it != dataMap.end());

	int keep_out[11070];
	int num_out;
	const float* dets = it->second->host_data();

	const uint32_t numBoxes = it->second->getShape(2);
	vector<vector<float>> dets1(numBoxes);
	vector<float> scores(numBoxes);

	for (uint32_t i = 0; i < numBoxes; i++) {
		dets1[i].resize(4);

		dets1[i][0] = dets[i*5+0];
		dets1[i][1] = dets[i*5+1];
		dets1[i][2] = dets[i*5+2];
		dets1[i][3] = dets[i*5+3];

		scores[i] = dets[i*5+4];
	}

	print2dArray("dets1", dets1);
	printArray("scores", scores);

	vector<uint32_t> keep;
	nms(dets1, scores, 0.7f, keep);

	printArray("keep", keep);


	/*
	for (uint32_t i = 0; i < 11070; i++) {
		for (uint32_t j = 0; j < 5; j++) {
			cout << dets[i*5+j] << ", ";
		}
		cout << endl;
	}
	_nms(keep_out, &num_out, dets, 11070, 5, 0.7f, 0);
	for (uint32_t i = 0; i < num_out; i++) {
		cout << keep_out[i] << endl;
	}
	*/
}



#endif




