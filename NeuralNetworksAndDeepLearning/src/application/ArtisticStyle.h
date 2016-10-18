/*
 * ArtisticStyle.h
 *
 *  Created on: 2016. 7. 22.
 *      Author: jhkim
 */

#ifndef ARTISTICSTYLE_H_
#define ARTISTICSTYLE_H_


//#ifndef GPU_MODE
#include "../network/Network.h"
#include "../debug/StatGraphPlotter.h"
#include <CImg.h>

using namespace cimg_library;


template <typename Dtype>
class ArtisticStyle {
public:
	ArtisticStyle()
	: contentRepLayers({""}),
	  styleRepLayers({""}),
	  contentReconstructionFactor(0),
	  styleReconstructionFactor(0),
	  learningRate(0),
	  end(""),
	  plotContentCost(false),
	  plotStyleCost(false) {}


	ArtisticStyle(Network<Dtype> *network,
			const string& contentImagePath,
			const string& styleImagePath,
			const vector<string>& contentRepLayers,
			const vector<string>& styleRepLayers,
			const float contentReconstructionFactor,
			const float styleReconstructionFactor,
			const float learningRate,
			const string& end,
			const bool plotContentCost,
			const bool plotStyleCost);
	virtual ~ArtisticStyle();
	void style();
	void test();

private:
	void computeContentRepresentationLayerResponses();
	void computeStyleRepresentationLayerResponses();

	//void computeContentLossGradient();
	//void computeStyleLossGradient();

	double computeContentLossGradientAt(const int contentLayerIndex);
	double computeStyleLossGradientAt(const int styleLayerIndex);


	//void computeContentLoss(uint32_t contentRepLayerIndex);
	//void computeStyleLoss(uint32_t styleRepLayerIndex);



	void preprocess(CImg<Dtype>* cimg);
	void preprocess(SyncMem<Dtype>* mem);
	void deprocess(CImg<Dtype>* cimg);
	void deprocess(SyncMem<Dtype>* mem);
	void clipImage(CImg<Dtype>* img);

	Data<Dtype>* createDataFromCImg(CImg<Dtype>* cimg);
	SyncMem<Dtype>* createGramMatrixFromData(Data<Dtype>* data);

	HiddenLayer<Dtype>* findRepresentationLayer(const string& layerName);

	SyncMem<Dtype>* flattenData(Data<Dtype>* data, const uint32_t height, const uint32_t width);
	void unflattenData(SyncMem<Dtype>* mem, const uint32_t flattenHeight, const uint32_t flattenWidth, Data<Dtype>* data);



	int findContentRepLayer(const string& layerName);
	int findStyleRepLayer(const string& layerName);



	void gramMatrixTest();
	void dataSubTest();
	void flattenTest();
	void unflattenTest();
	void gemmTest();

	void on() { Data<Dtype>::printConfig = 1; }
	void off() { Data<Dtype>::printConfig = 0; }



private:
	Network<Dtype>* network;
	CImg<Dtype>* x;			// input image
	CImg<Dtype>* p;			// photo, content image
	CImg<Dtype>* a;			// art, style image

	Data<Dtype>* xdata;
	//Data<Dtype>* pdata;
	//Data<Dtype>* adata;

	CImgDisplay* xdisp;
	CImgDisplay* pdisp;
	CImgDisplay* adisp;

	uint32_t width;
	uint32_t height;
	uint32_t channel;

	const vector<string> contentRepLayers;
	const vector<string> styleRepLayers;

	vector<SyncMem<Dtype>*> contentRepLayerResps;		// feature map
	vector<SyncMem<Dtype>*> styleRepLayerResps;			// gram matrix of feature map

	SyncMem<Dtype> contentGrad;
	SyncMem<Dtype> styleGrad;

	const float contentReconstructionFactor;
	const float styleReconstructionFactor;

	const float learningRate;

	//float mean[3];

	const string end;


	StatGraphPlotter contentCostLogger;
	StatGraphPlotter styleCostLogger;
	SyncMem<Dtype> mean;
	SyncMem<Dtype> xdataTemp;
	CImg<Dtype>* xTemp;
	//CImgDisplay* xTempDisp;

	const bool plotContentCost;
	const bool plotStyleCost;
};

//#endif

#endif /* ARTISTICSTYLE_H_ */
