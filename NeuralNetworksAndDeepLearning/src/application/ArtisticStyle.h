/*
 * ArtisticStyle.h
 *
 *  Created on: 2016. 7. 22.
 *      Author: jhkim
 */

#ifndef ARTISTICSTYLE_H_
#define ARTISTICSTYLE_H_


//#ifndef GPU_MODE

#include "common.h"
#include "Network.h"
#include "StatGraphPlotter.h"

#include <CImg.h>
//using namespace cimg_library;


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
			const std::string& contentImagePath,
			const std::string& styleImagePath,
			const std::vector<std::string>& contentRepLayers,
			const std::vector<std::string>& styleRepLayers,
			const float contentReconstructionFactor,
			const float styleReconstructionFactor,
			const float learningRate,
			const std::string& end,
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


	/**
	 * 이미지에 대해 mean값 subtract
	 */
	void preprocess(cimg_library::CImg<Dtype>* cimg);
	void preprocess(SyncMem<Dtype>* mem);
	void deprocess(cimg_library::CImg<Dtype>* cimg);
	void deprocess(SyncMem<Dtype>* mem);
	void clipImage(cimg_library::CImg<Dtype>* img);

	void createDataFromCImg(cimg_library::CImg<Dtype>* cimg, Data<Dtype>* data);
	void createGramMatrixFromData(Data<Dtype>* data, SyncMem<Dtype>* gramMatrix);

	HiddenLayer<Dtype>* findRepresentationLayer(const std::string& layerName);

	void flattenData(Data<Dtype>* data, const uint32_t height, const uint32_t width, SyncMem<Dtype>* mem);
	void unflattenData(SyncMem<Dtype>* mem, const uint32_t flattenHeight, const uint32_t flattenWidth, Data<Dtype>* data);



	int findContentRepLayer(const std::string& layerName);
	int findStyleRepLayer(const std::string& layerName);



	void gramMatrixTest();
	void dataSubTest();
	void flattenTest();
	void unflattenTest();
	void gemmTest();

	void on() { Data<Dtype>::printConfig = 1; }
	void off() { Data<Dtype>::printConfig = 0; }


	void feedforwardWithData(Data<Dtype>* data);




	void updateDimensionFromCImg(cimg_library::CImg<Dtype>* cimg);
	void prepareCImgFromPath(const std::string& path, cimg_library::CImg<Dtype>*& cimg);
	void normalizeCImg(cimg_library::CImg<Dtype>* cimg);
	void updateMean();
	void prepareCImgDisplay(cimg_library::CImg<Dtype>* cimg, const std::string& title,
			cimg_library::CImgDisplay*& cimgDisplay);
	bool isValidImageDimension(cimg_library::CImg<Dtype>* cimg);






	void printCImg(cimg_library::CImg<Dtype>* cimg);



private:
	Network<Dtype>* network;

	cimg_library::CImg<Dtype>* x;			// input image
	cimg_library::CImg<Dtype>* p;			// photo, content image
	cimg_library::CImg<Dtype>* a;			// art, style image
	cimg_library::CImg<Dtype>* xTemp;

	cimg_library::CImgDisplay* xdisp;
	cimg_library::CImgDisplay* pdisp;
	cimg_library::CImgDisplay* adisp;

	Data<Dtype>* xdata;
	SyncMem<Dtype> xdataTemp;
	SyncMem<Dtype> mean;

	uint32_t width;
	uint32_t height;
	uint32_t channel;

	const std::vector<std::string> contentRepLayers;
	const std::vector<std::string> styleRepLayers;
    std::vector<SyncMem<Dtype>*> contentRepLayerResps;		// feature map
    std::vector<SyncMem<Dtype>*> styleRepLayerResps;		// gram matrix of feature map

	const float contentReconstructionFactor;
	const float styleReconstructionFactor;
	const float learningRate;
	const std::string end;

	StatGraphPlotter contentCostLogger;
	StatGraphPlotter styleCostLogger;

	const bool plotContentCost;
	const bool plotStyleCost;
};

//#endif

#endif /* ARTISTICSTYLE_H_ */
