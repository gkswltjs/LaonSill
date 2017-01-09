#define BBOXTRANSFORMUTIL_TEST 0

#if BBOXTRANSFORMUTIL_TEST



#include "LayerConfig.h"
#include "test_common.h"
#include "BboxTransformUtil.h"
#include "Cuda.h"


using namespace std;



int main() {

	Data<float>::printConfig = true;


	// boxes1 data
	param_filler<float> data1Filler;
	data1Filler.type = ParamFillerType::Gaussian;
	data1Filler.value = 0.1f;

	Data<float>* boxes1Data = new Data<float>("boxes1");
	boxes1Data->reshape({1, 1, 128, 4});
	data1Filler.fill(boxes1Data);
	boxes1Data->print_data({}, false);

	vector<vector<float>> boxes1;
	fill2dVecWithData(boxes1Data, boxes1);
	print2dArray("boxes1", boxes1);


	// boxes2 data
	param_filler<float> data2Filler;
	data2Filler.type = ParamFillerType::Gaussian;
	data2Filler.value = 0.09f;

	Data<float>* boxes2Data = new Data<float>("boxes2");
	boxes2Data->reshape({1, 1, 128, 4});
	data2Filler.fill(boxes2Data);
	boxes2Data->print_data({}, false);

	vector<vector<float>> boxes2;
	fill2dVecWithData(boxes2Data, boxes2);
	print2dArray("boxes2", boxes2);


	// compute bbox pred
	vector<vector<float>> bboxPred(boxes1.size());
	for (uint32_t i = 0; i < boxes1.size(); i++)
		bboxPred[i].resize(4);

	BboxTransformUtil::bboxTransform(boxes1, 0, boxes2, 0, bboxPred, 0);
	print2dArray("bboxPred", bboxPred);

	Data<float>* bboxPredData = new Data<float>("bboxPred");
	bboxPredData->reshape({1, 1, 128, 4});
	bboxPredData->fill_host_with_2d_vec(bboxPred, {0, 1, 2, 3});
	bboxPredData->print_data({}, false);


	// compute pred bbox
	vector<vector<float>> predBbox;
	BboxTransformUtil::bboxTransformInv(boxes1, bboxPredData, predBbox);
	print2dArray("predBbox", predBbox);

	Data<float>::printConfig = false;

	delete boxes1Data;
	delete boxes2Data;
	delete bboxPredData;
}


#endif




