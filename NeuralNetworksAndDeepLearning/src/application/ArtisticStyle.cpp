/*
 * ArtisticStyle.cpp
 *
 *  Created on: 2016. 7. 22.
 *      Author: jhkim
 */


//#ifndef GPU_MODE


#include <stdint.h>
#include <vector>

#include "ArtisticStyle.h"
#include "Data.h"
#include "InputLayer.h"
#include "ApplicationCudaFunctions.h"
#include "cnpy.h"

using namespace std;
using namespace cimg_library;


#define SMALL_TEST 0

#if !SMALL_TEST
#define ORIG_DIM	224
#define LENGTH		512
#else
#define ORIG_DIM	16
#define LENGTH		16
#endif
#define STYLE_SCALE	1.2
#define RATIO		10000

#define ARTISTICSTYLE_LOG	0
//#define CONTENT_
#define STYLE_


/**
 * CImg
 * (1) R, G, B 채널순으로 채널이 분리되어 데이터가 들어있음
 * 		[R,R,R,R...,R]
 * 		[G,G,G,G,..,G]
 * 		[B,B,B,B,..,B]
 * (2) cimg의 normalize() 함수 사용시, 이미지의 최대, 최소값을 기준으로 normalize하므로 주의
 * 		(사실 이러한 방식의 normalize는 사용할 수 없음)
 *
 */


template <typename Dtype>
ArtisticStyle<Dtype>::ArtisticStyle(Network<Dtype> *network,
		const string& contentImagePath,
		const string& styleImagePath,
		const vector<string>& contentRepLayers,
		const vector<string>& styleRepLayers,
		const float contentReconstructionFactor,
		const float styleReconstructionFactor,
		const float learningRate,
		const string& end,
		const bool plotContentCost,
		const bool plotStyleCost)
		: network(network),
		  contentRepLayers(contentRepLayers),
		  styleRepLayers(styleRepLayers),
		  contentReconstructionFactor(contentReconstructionFactor),
		  styleReconstructionFactor(styleReconstructionFactor),
		  learningRate(learningRate),
		  end(end),
		  contentCostLogger(100, 10),
		  styleCostLogger(100, 10),
		  styleUnitCostLogger(100, 10),
		  plotContentCost(plotContentCost),
		  plotStyleCost(plotStyleCost) {

	this->lrMult = 1.0;

	// ****************************************************************
	// for dataMean
	// (1) ImageNet의 data mean값을 설정
	// ****************************************************************
	this->dataMean.reshape(3);
	Dtype* hDataMean = this->dataMean.mutable_host_mem();
	hDataMean[0] = Dtype(0.47684615850);
	hDataMean[1] = Dtype(0.45469805598),
	hDataMean[2] = Dtype(0.41394191980);
	//hDataMean[0] = Dtype(122.67891434);
	//hDataMean[1] = Dtype(116.66876762),
	//hDataMean[2] = Dtype(104.00698793);

	// compute data bounds
	this->dataMin.reshape(3);
	this->dataMax.reshape(3);
	Dtype* hDataMin = this->dataMin.mutable_host_mem();
	Dtype* hDataMax = this->dataMax.mutable_host_mem();

	for (int i = 0; i < 3; i++) {
		hDataMin[i] = -this->dataMean.host_mem()[i];
		hDataMax[i] = hDataMin[i] + 1.0f;
	}


	// ****************************************************************
	// for photo, content image "p"
	// (1) load content image to CImg from contentImagePath
	// (2) dispaly content image
	// (3) update image dimension from content image
	// (4) fill mean according to the updated dimension
	// ****************************************************************
	prepareCImgFromPath(contentImagePath, this->p);
	printCImg(this->p, "content image", false);
	float pScale = max(LENGTH / float(max(this->p->width(), this->p->height())),
			ORIG_DIM / float(min(this->p->width(), this->p->height())));
	this->p->resize(this->p->width() * pScale, this->p->height() * pScale, -100, -100, 3);
	cout << "resize scale: " << pScale << endl;
	printCImg(this->p, "resized content image", false);
	prepareCImgDisplay(this->p, "content", this->pdisp);
	updateDimensionFromCImg(this->p);
	fillMean();			// dimension 정보 업데이트 후, 해당 정보 기반  mean값 업데이트.


	// ****************************************************************
	// for art, style image "a"
	// (1) load style image to CImg from styleImagePath
	// (2) display style image
	// ****************************************************************
	prepareCImgFromPath(styleImagePath, this->a);
	printCImg(this->a, "style image", false);
	float aScale = max(LENGTH / float(max(this->a->width(), this->a->height())),
			ORIG_DIM / float(min(this->a->width(), this->a->height())));
	//aScale *= STYLE_SCALE;
	this->a->resize(this->a->width() * aScale, this->a->height() * aScale, -100, -100, 3);
	cout << "resize scale: " << aScale << endl;
	printCImg(this->a, "resized style image", false);
	prepareCImgDisplay(this->a, "style", this->adisp);
	assert(isValidImageDimension(this->a));


	// ****************************************************************
	// for input image "x"
	// (1) xdata 객체만 생성하고 구체적인 초기화는 style()에서 수행
	// ****************************************************************
	this->xdata = new Data<Dtype>("xdata");

	// ****************************************************************
	// for temp input image x (display purpose)
	// (1) optimizing 도중의 xdata값 출력을 위한 임시의 xdataTemp를 초기화
	// (2) x와 xdataTemp간의 데이터를 공유하여
	//			xdata ->(copy)-> xdataTemp -> x -> xdisp 과정을 통해 화면 출력
	// ****************************************************************
	this->xdataTemp.reshape(this->channel * this->height * this->width);
	this->x = new CImg<Dtype>(this->xdataTemp.mutable_host_mem(), this->width, this->height,
			1, this->channel, true);
	prepareCImgDisplay(this->x, "input", this->xdisp);


	const uint32_t numStyleRepLayers = styleRepLayers.size();
	this->styleLossScales.resize(numStyleRepLayers);
	this->styleLossWeights.resize(numStyleRepLayers);
	for (uint32_t i = 0; i < numStyleRepLayers; i++) {
		this->styleLossScales[i] = 1.0;
		this->styleLossWeights[i] = 1.0 / numStyleRepLayers;
	}
}


template <typename Dtype>
void ArtisticStyle<Dtype>::updateDimensionFromCImg(CImg<Dtype>* cimg) {
	this->width		= cimg->width();
	this->height 	= cimg->height();
	this->channel 	= cimg->spectrum();
}

template <typename Dtype>
void ArtisticStyle<Dtype>::prepareCImgFromPath(const string& path,
		cimg_library::CImg<Dtype>*& cimg) {
	cimg = new CImg<Dtype>(path.c_str());
}


template <typename Dtype>
void ArtisticStyle<Dtype>::fillMean() {
	const uint32_t n = this->channel * this->height * this->width;
	const uint32_t singleChannelSize = this->height * this->width;

	this->mean.reshape(n);

	fill_channel_mean(n, singleChannelSize, this->dataMean.device_mem(),
			this->mean.mutable_device_mem());

	//on();
	//this->mean.print("mean", {1, this->channel, this->height, this->width}, false);
	//off();
}

template <typename Dtype>
void ArtisticStyle<Dtype>::prepareCImgDisplay(CImg<Dtype>* cimg, const string& title,
		CImgDisplay*& cimgDisplay) {
	cimgDisplay = new CImgDisplay(*cimg, title.c_str());
}

template <typename Dtype>
bool ArtisticStyle<Dtype>::isValidImageDimension(CImg<Dtype>* cimg) {
	if (cimg->width() <= 0 ||
			cimg->height() <= 0 ||
			cimg->spectrum() <= 0 ||
			this->width != cimg->width() ||
			this->height != cimg->height() ||
			this->channel != cimg->spectrum()) {
		return false;
	}
	return true;
}

template <typename Dtype>
ArtisticStyle<Dtype>::~ArtisticStyle() {
	if (this->network) delete this->network;

	if (this->x) delete this->x;
	if (this->p) delete this->p;
	if (this->a) delete this->a;

	if (this->xdisp) delete this->xdisp;
	if (this->pdisp) delete this->pdisp;
	if (this->adisp) delete this->adisp;

	if (this->xdata) delete this->xdata;

	for (uint32_t i = 0; i < this->contentRepLayerResps.size(); i++)
		delete this->contentRepLayerResps[i];
	this->contentRepLayerResps.clear();

	for (uint32_t i = 0; i < this->styleRepLayerResps.size(); i++)
		delete this->styleRepLayerResps[i];
	this->styleRepLayerResps.clear();
}



template <typename Dtype>
void ArtisticStyle<Dtype>::style() {
#ifdef CONTENT_
	// compute content representations
	computeContentRepresentationLayerResponses();
#endif
#ifdef STYLE_
	// compute style representations
	computeStyleRepresentationLayerResponses();
#endif

	// generate initial net input
	this->xdata->reshape({1, this->channel, this->height, this->width});
	makeNoiseInput(this->xdata);

	LayersConfig<Dtype>* layersConfig = this->network->getLayersConfig();
	InputLayer<Dtype>* inputLayer = layersConfig->_inputLayer;

	uint32_t updateCnt = 0;
	double contentCost = 0.0;
	double styleCost = 0.0;

	while(true) {
		contentCost = 0.0;
		styleCost = 0.0;

		feedforwardWithData(this->xdata);
		for (int i = layersConfig->_layers.size()-1; i > 0; i--) {
			Layer<Dtype>* repLayer =
					dynamic_cast<Layer<Dtype>*>(layersConfig->_layers[i]);
			assert(repLayer);

			int repLayerIndex = -1;
#ifdef CONTENT_
			if ((repLayerIndex = findContentRepLayer(repLayer->getName())) >= 0)
				contentCost += computeContentLossGradientAt(repLayerIndex);
#endif
#ifdef STYLE_
			if (repLayerIndex == -1 &&
					((repLayerIndex = findStyleRepLayer(repLayer->getName())) >= 0)) {
				double unitCost = computeStyleLossGradientAt(repLayerIndex);

				styleCost += unitCost;
				if (this->plotStyleCost) {
					this->styleUnitCostLogger.addStat(this->styleRepLayers.size()-1-repLayerIndex,
							repLayer->name, unitCost);
				}
			}

#endif
			repLayer->backpropagation();
		}

#ifdef CONTENT_
		if (this->plotContentCost)
			this->contentCostLogger.addStat(0, "content_cost", contentCost);
#endif
#ifdef STYLE_
		if (this->plotStyleCost)
			this->styleCostLogger.addStat(0, "style_cost", styleCost);
#endif

		updateCnt++;
		if (updateCnt % 5000 == 0) {
			cout << "lrMult is updated from " << this->lrMult;
			this->lrMult /= 2;
			cout << " to " << this->lrMult << endl;
		}


		//on();
		//inputLayer->_outputData[0]->print_grad({}, false);
		//off();




		float lr = -1.0f * this->learningRate * this->lrMult;
		const Dtype* grad_device = inputLayer->_outputData[0]->device_grad();
		Dtype* xdata_device = this->xdata->mutable_device_data();
		checkCudaErrors(cublasSaxpy(Cuda::cublasHandle,
				static_cast<int>(this->xdata->getCount()),
                &lr, grad_device, 1, xdata_device, 1));

		const Dtype* dDataMin = this->dataMin.device_mem();
		const Dtype* dDataMax = this->dataMax.device_mem();
		boundData(dDataMin, dDataMax, this->xdata);

		if (updateCnt % 1 == 0) {
			this->xdataTemp.set_mem(this->xdata->device_data(),
					SyncMemCopyType::DeviceToDevice, 0, this->xdata->getCount());
			deprocess(&this->xdataTemp);
			// xdataTemp의 host mem으로 copy해서 가져온다.
			this->xdataTemp.host_mem();
			this->xdisp->resize(*this->x, true).display(*this->x);
		}
		//while (!this->xdisp->is_key()) { this->xdisp->wait(); }
	}
	while (!this->xdisp->is_closed()) { this->xdisp->wait(); }
	while (!this->pdisp->is_closed()) { this->pdisp->wait(); }
}




template <typename Dtype>
void ArtisticStyle<Dtype>::boundData(const Dtype* dataMin, const Dtype* dataMax,
		Data<Dtype>* data) {

	const uint32_t n = data->getCount();
	const uint32_t singleChannelSize = data->getCountByAxis(2);

	//on();
	//data->print_data({}, false);
	//off();

	bound_data(n, singleChannelSize, dataMin, dataMax, data->mutable_device_data());

	//on();
	//data->print_data({}, false);
	//off();
}


template <typename Dtype>
void ArtisticStyle<Dtype>::makeNoiseInput(Data<Dtype>* input) {
	float inputScale = 0.0f;
	float noiseScale = 1.0f - inputScale;

	// CONTENT
	input->reshape({1, this->channel, this->height, this->width});
	input->reset_device_data();
	//createDataFromCImg(this->p, input);		// 255 scale
	//input->scale_device_data(inputScale);


	// RANDOM PINK NOISE
	CImg<Dtype>* noise = 0;
#if !SMALL_TEST
	string path = "/home/jkim/x_"+to_string(LENGTH)+"_1.png";
#else
	string path = "/home/jkim/x_"+to_string(LENGTH)+"_1.jpg";
#endif
	cout << "noise image: " << path << endl;
	prepareCImgFromPath(path, noise);
	Data<Dtype>* temp = new Data<Dtype>("temp");
	createDataFromCImg(noise, temp);		// 255 scale

	temp->scale_device_data(noiseScale);
	input->add_device_data(temp);
	//input->scale_device_data(0.5f);



	/*
	this->xdataTemp.set_mem(input->device_data(),
			SyncMemCopyType::DeviceToDevice, 0, input->getCount());
	this->xdataTemp.host_mem();
	this->xdisp->resize(*this->x, true).display(*this->x);
	while (!this->xdisp->is_key()) { this->xdisp->wait(); }
	*/



	/*
	// XXX: neural-style에 따라
	// GAUSSIAN
	param_filler<Dtype> noiseFiller(ParamFillerType::Gaussian, 1.0);
	noiseFiller.fill(input);

	// neural-scale에서 256 level의 input image에 대해 0.256으로 scale down하는 것으로 추정
	// for now, 0.0~1.0 scale이지만 preprocess에서 scale다운하므로 여기서는 scale 그대로 따름.
	input->scale_device_data(0.256);
	*/


	//on();
	//input->print_data({}, false);
	//off();




	preprocess(input->_data.get());

	if (noise) delete noise;
	if (temp) delete temp;

}



















template <typename Dtype>
void ArtisticStyle<Dtype>::computeContentRepresentationLayerResponses() {
	Data<Dtype>* pdata = new Data<Dtype>("pdata");
	createDataFromCImg(this->p, pdata);
	preprocess(pdata->_data.get());

	feedforwardWithData(pdata);
	delete pdata;

	// feedforward() 결과, 지정된 레이어의 response를 복사, 보관 (Content Loss계산용)
	const uint32_t numContentRepLayers = this->contentRepLayers.size();
	this->contentRepLayerResps.clear();
	this->contentRepLayerResps.resize(numContentRepLayers);

	for(uint32_t i = 0; i < numContentRepLayers; i++) {
		Layer<Dtype>* contentRepLayer = findRepresentationLayer(this->contentRepLayers[i]);
		cout << "for content, found: " << contentRepLayer->name << endl;

		SyncMem<Dtype>* contentRepLayerResp =
            new SyncMem<Dtype>(contentRepLayer->_outputData[0]->_data.get());


#if SMALL_TEST
		on();
		contentRepLayer->_outputData[0]->print_data({}, true);
		contentRepLayerResp->print("contentRepLayerResp",
				contentRepLayer->_outputData[0]->getShape(), true);
		off();
#endif

		this->contentRepLayerResps[i] = contentRepLayerResp;
	}
}

template <typename Dtype>
void ArtisticStyle<Dtype>::computeStyleRepresentationLayerResponses() {
	Data<Dtype>* adata = new Data<Dtype>("adata");
	createDataFromCImg(this->a, adata);
	preprocess(adata->_data.get());

	feedforwardWithData(adata);
	delete adata;

	// feedforward() 결과, 지정된 레이어의 response를 복사, 보관 (Style Loss계산용)
	const uint32_t numStyleRepLayers = this->styleRepLayers.size();
	this->styleRepLayerResps.clear();
	this->styleRepLayerResps.resize(numStyleRepLayers);

	//cout << "StyleRepresentationLayerResponseInfo:" << endl;
	for(uint32_t i = 0; i < numStyleRepLayers; i++) {
		Layer<Dtype>* styleRepLayer = findRepresentationLayer(styleRepLayers[i]);
		SyncMem<Dtype>* styleRepLayerResp = new SyncMem<Dtype>();
        createGramMatrixFromData(styleRepLayer->_outputData[0], 0, styleRepLayerResp);


        /*
        on();
        const uint32_t nl = styleRepLayer->_outputData[0]->channels();
        styleRepLayer->_outputData[0]->print_data({}, false);
        styleRepLayerResp->print("styleRepLayerResp", {1, 1, nl, nl}, true);
        off();
        */


		this->styleRepLayerResps[i] = styleRepLayerResp;
	}
}

template <typename Dtype>
void ArtisticStyle<Dtype>::feedforwardWithData(Data<Dtype>* data) {
	LayersConfig<Dtype>* layersConfig = this->network->getLayersConfig();
	InputLayer<Dtype>* inputLayer = layersConfig->_inputLayer;

	// feedforward에 의해 reshape 전이기 때문에 inputData가 초기화되기 전,
	// 반드시 outputData에 데이터를 설정해줘야 한다.
	inputLayer->_outputData[0]->reshapeLike(data);
	inputLayer->_outputData[0]->set_device_data(data);
	for (uint32_t i = 0; i < layersConfig->_layers.size(); i++) {
		layersConfig->_layers[i]->feedforward();
		if (layersConfig->_layers[i]->name == this->end)
			break;
	}
}











template <typename Dtype>
double ArtisticStyle<Dtype>::computeContentLossGradientAt(const int contentLayerIndex) {
	const string& contentRepLayerName = this->contentRepLayers[contentLayerIndex];

	// index에 해당하는 content representation layer
	Layer<Dtype>* contentRepLayer = findRepresentationLayer(contentRepLayerName);
	//cout << "compute content loss gradient at " << contentLayerIndex << ", " <<
	//		contentRepLayer->name << endl;

	SyncMem<Dtype>* presp = this->contentRepLayerResps[contentLayerIndex];
	SyncMem<Dtype>* xresp = contentRepLayer->_outputData[0]->_data.get();

#if SMALL_TEST
	on();
	const vector<uint32_t> shape = contentRepLayer->_outputData[0]->getShape();
	presp->print("presp", shape, true);
	xresp->print("xresp", shape, true);
	off();
#endif

	// add하는 케이스 때문에 임시 SyncMem 객체를 사용
	SyncMem<Dtype>* contentLoss = new SyncMem<Dtype>();
	contentLoss->reshape(contentRepLayer->_outputData[0]->_grad->getSize());

	const uint32_t N = xresp->getSize();
	diff_content_loss(N, xresp->device_mem(), presp->device_mem(),
			contentLoss->mutable_device_mem());

#if SMALL_TEST
	on();
	contentLoss->print("contentLoss", shape, true);
	off();
#endif

	//if (this->contentReconstructionFactor != 1.0f) {
		float scale = this->contentReconstructionFactor / N;
		cout << "scale for " << contentRepLayer->name << " is " << scale << endl;
		contentLoss->scale_device_mem(scale);
	//}


	if (contentRepLayerName == this->end) {
		contentRepLayer->_outputData[0]->_grad->set_mem(contentLoss->device_mem(),
				SyncMemCopyType::DeviceToDevice, 0, contentLoss->getSize());
	} else
		contentRepLayer->_outputData[0]->_grad->add_device_mem(contentLoss->device_mem());

	double contentCost = 0.5*contentLoss->sumsq_device_mem();
	delete contentLoss;

	return contentCost;
}

template <typename Dtype>
double ArtisticStyle<Dtype>::computeStyleLossGradientAt(const int styleLayerIndex) {
	const string& styleRepLayerName = this->styleRepLayers[styleLayerIndex];

	Layer<Dtype>* styleRepLayer = findRepresentationLayer(styleRepLayerName);
	assert(styleRepLayer);

	const uint32_t nl = styleRepLayer->_outputData[0]->channels();
	const uint32_t ml = styleRepLayer->_outputData[0]->height() *
			styleRepLayer->_outputData[0]->width();
	const double c = 1.0 / (1.0 * nl * nl * ml * ml);

	SyncMem<Dtype>* Fl = new SyncMem<Dtype>();
	SyncMem<Dtype>* Gl = new SyncMem<Dtype>();	//xresp
	createGramMatrixFromData(styleRepLayer->_outputData[0], Fl, Gl);








	SyncMem<Dtype>* G_style = this->styleRepLayerResps[styleLayerIndex];
	Gl->sub_device_mem(G_style->device_mem());

	SyncMem<Dtype>* gemmResult = new SyncMem<Dtype>();
		gemmResult->reshape(nl*ml);

	// 8. Gl = trans(trans(Fl)*Gl)
	const Dtype* xresp_device = Gl->device_mem();
	const Dtype* f_device = Fl->device_mem();
	Dtype* gemmResult_device = gemmResult->mutable_device_mem();
	checkCudaErrors(cublasSgemm(Cuda::cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
			nl, ml, nl,
			&Cuda::alpha, xresp_device, nl, f_device, nl,
			&Cuda::beta, gemmResult_device, nl));

	const uint32_t N = Fl->getSize();
	diff_style_loss(N, f_device, gemmResult_device);

	// 9. Gl = 1/(nl*nl*ml*ml)*Gl
	Data<Dtype>* gemmResultd = new Data<Dtype>("gemmResult",
			styleRepLayer->_outputData[0]->getShape());
	unflattenData(gemmResult, nl, ml, gemmResultd);

	const double a = this->styleReconstructionFactor *
			this->styleLossWeights[styleLayerIndex] *
			this->styleLossScales[styleLayerIndex];
	//double scaleFactor =  a * c;


	double scaleFactor = 1.0 / G_style->getSize() * this->styleReconstructionFactor;

	/*
	streamsize ss = cout.precision();
	cout.precision(20);
	cout << "for " << styleRepLayerName <<  ", a: " << a << ", b: " << c <<
			", scaleFactor: " << scaleFactor << endl;
	cout.precision(ss);
	*/

	gemmResultd->scale_device_data(scaleFactor);

	cout << "scale for " << styleRepLayer->name << " is " << scaleFactor << endl;

	//on();
	//gemmResultd->print_data({}, true);
	//off();

	if (styleRepLayerName == this->end) {
		styleRepLayer->_outputData[0]->_grad->set_mem(gemmResultd->_data->device_mem(),
				SyncMemCopyType::DeviceToDevice, 0, gemmResultd->_data->getSize());
	} else
		styleRepLayer->_outputData[0]->_grad->add_device_mem(gemmResultd->_data->device_mem());

	double styleCost = Gl->sumsq_device_mem();
	styleCost = 0.5 * scaleFactor * styleCost;

	delete gemmResult;
	delete gemmResultd;
	delete Gl;
	delete Fl;

	return styleCost;
}

template <typename Dtype>
void ArtisticStyle<Dtype>::createDataFromCImg(CImg<Dtype>* cimg, Data<Dtype>* data) {
	data->reshape({1, (uint32_t)cimg->spectrum(), (uint32_t)cimg->height(),
		(uint32_t)cimg->width()});
	data->set_device_with_host_data(cimg->data(), 0, cimg->size());
}


template <typename Dtype>
void ArtisticStyle<Dtype>::createGramMatrixFromData(Data<Dtype>* data,
		SyncMem<Dtype>* Fl, SyncMem<Dtype>* gramMatrix) {
	const uint32_t nl = data->channels();					// number of feature maps
	const uint32_t ml = data->height()*data->width();		// number of feature map elements
	gramMatrix->reshape(nl*nl);

	SyncMem<Dtype>* tempFl = 0;
	if (Fl)
		tempFl = Fl;
	else
		tempFl = new SyncMem<Dtype>();
	flattenData(data, nl, ml, tempFl);

	const Dtype* fdata_device = tempFl->device_mem();
	Dtype* gram_device = gramMatrix->mutable_device_mem();

	checkCudaErrors(cublasSgemm(Cuda::cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
			nl, nl, ml,
			&Cuda::alpha, fdata_device, nl, fdata_device, nl,
			&Cuda::beta, gram_device, nl));

	// XXX: neural-style에 따라 normalize
	float scale = 1.0 / tempFl->getSize();
	gramMatrix->scale_device_mem(scale);

	if (!Fl)
		delete tempFl;
}


template <typename Dtype>
Layer<Dtype>* ArtisticStyle<Dtype>::findRepresentationLayer(const string& layerName) {
	Layer<Dtype>* repLayer =
			dynamic_cast<Layer<Dtype>*>(network->findLayer(layerName));
	if(!repLayer) {
		cout << layerName << " specified representation layer is not hidden layer ... " <<
				endl;
		exit(1);
	}
	return repLayer;
}





template <typename Dtype>
void ArtisticStyle<Dtype>::flattenData(Data<Dtype>* data, const uint32_t flattenHeight,
		const uint32_t flattenWidth, SyncMem<Dtype>* mem) {

	// 헷갈리니까 개념적 말고 물리적 도메인으로 몰아넣자.
	// physical memory domain을 기준으로 생각하자.
	// p: physical, f: flatten, o: original
	const uint32_t pfh = flattenWidth;
	const uint32_t pfw = flattenHeight;
	const uint32_t pfs = pfh*pfw;

	const uint32_t poc = data->channels();		// pfw
	const uint32_t poh = data->width();
	const uint32_t pow = data->height();
	const uint32_t poHxW = poh*pow;				// pfh

	if(pfs != data->getCount() ||
			pfw != poc ||
			pfh != poHxW) {
		cout << "invalid flatten shape ... " << endl;
		exit(1);
	}
	mem->reshape(pfs);

	const Dtype* data_host = data->host_data();
	Dtype* mem_host = mem->mutable_host_mem();

	for(uint32_t w = 0; w < pow; w++) {
		for(uint32_t h = 0; h < poh; h++) {
			for(uint32_t c = 0; c < poc; c++) {
				mem_host[c+(h+w*poh)*pfw] = data_host[c*poHxW+h*pow+w];
			}
		}
	}
}



/*
template <typename Dtype>
void ArtisticStyle<Dtype>::flattenData(Data<Dtype>* data, const uint32_t flattenHeight,
		const uint32_t flattenWidth, SyncMem<Dtype>* mem) {

	// 헷갈리니까 개념적 말고 물리적 도메인으로 몰아넣자.
	// physical memory domain을 기준으로 생각하자.
	// p: physical, f: flatten, o: original
	const uint32_t pfh = flattenWidth;
	const uint32_t pfw = flattenHeight;
	const uint32_t pfs = pfh*pfw;

	const uint32_t poc = data->channels();		// pfw
	const uint32_t poh = data->width();
	const uint32_t pow = data->height();
	const uint32_t poHxW = poh*pow;				// pfh

	assert(flattenHeight == poc);
	assert(flattenWidth == poHxW);
	assert(pfs == data->getCount());

	mem->reshape(pfs);

	const Dtype* data_host = data->host_data();
	Dtype* mem_host = mem->mutable_host_mem();

	for (uint32_t c = 0; c < poc; c++) {
		// w방향이 개념적인 열에 대한 정보
		for (uint32_t w = 0; w < pow; w++) {
			// h방향이 개념적인 행에 대한 정보
			for (uint32_t h = 0; h < poh; h++) {

				mem_host[c * pfh + w * poh + h] =
						// c는 개념적으로나 물리적으로나 동일!
						// h changes first!
						data_host[c * poHxW + w * poh + h];
			}
		}
	}
}
*/


template <typename Dtype>
void ArtisticStyle<Dtype>::unflattenData(SyncMem<Dtype>* mem, const uint32_t flattenHeight,
		const uint32_t flattenWidth, Data<Dtype>* data) {
	const uint32_t pfh = flattenWidth;
	const uint32_t pfw = flattenHeight;
	const uint32_t pfs = pfh*pfw;

	const uint32_t poc = data->channels();		// pfw
	const uint32_t poh = data->width();
	const uint32_t pow = data->height();
	const uint32_t poHxW = poh*pow;				// pfh

	if(pfs != data->getCount() ||
			pfw != poc ||
			pfh != poHxW) {
		cout << "invalid flatten shape ... " << endl;
		exit(1);
	}

	const Dtype* mem_host = mem->mutable_host_mem();
	Dtype* data_host = data->mutable_host_data();

	for(uint32_t w = 0; w < pow; w++) {
		for(uint32_t h = 0; h < poh; h++) {
			for(uint32_t c = 0; c < poc; c++) {
				data_host[c*poHxW+h*pow+w] = mem_host[c+(h+w*poh)*pfw];
			}
		}
	}
}

template <typename Dtype>
void ArtisticStyle<Dtype>::preprocess(SyncMem<Dtype>* mem) {
	// scale
	const float scale = 1.0f / 255.0f;
	mem->scale_device_mem(scale);

	// subtract mean
	mem->sub_device_mem(this->mean.device_mem());

}

template <typename Dtype>
void ArtisticStyle<Dtype>::deprocess(SyncMem<Dtype>* mem) {
	//
	mem->add_device_mem(this->mean.device_mem());

	const float scale = 255.0f;
	mem->scale_device_mem(scale);
}







template <typename Dtype>
int ArtisticStyle<Dtype>::findContentRepLayer(const string& layerName) {
	const uint32_t numContentRepLayers = contentRepLayers.size();
	for(uint32_t i = 0; i < numContentRepLayers; i++) {
		if(layerName == contentRepLayers[i]) {
			return i;
		}
	}
	return -1;
}

template <typename Dtype>
int ArtisticStyle<Dtype>::findStyleRepLayer(const string& layerName) {
	const uint32_t numStyleRepLayers = styleRepLayers.size();
	for(uint32_t i = 0; i < numStyleRepLayers; i++) {
		if(layerName == styleRepLayers[i]) {
			return i;
		}
	}
	return -1;
}











template <typename Dtype>
void ArtisticStyle<Dtype>::gramMatrixTest() {
	const uint32_t numFeatureMaps = 5;
	Data<Dtype>* data = new Data<Dtype>("data");
	data->reshape({1, numFeatureMaps, 4, 4});

	param_filler<Dtype> weight_filler(ParamFillerType::Xavier, 0.1);
	weight_filler.fill(data);

	//on();
	//data->print_data("data:");
	//off();


	SyncMem<Dtype>* gramMatrix = new SyncMem<Dtype>();
	createGramMatrixFromData(data, 0, gramMatrix);
	gramMatrix->print("gramMatrix:");
	gramMatrix->print("gramMatrix:", {1, 1, numFeatureMaps, numFeatureMaps});

	delete gramMatrix;
}

template <typename Dtype>
void ArtisticStyle<Dtype>::dataSubTest() {
	on();

	Data<Dtype>* data1 = new Data<Dtype>("data1");
	data1->reshape({1, 3, 4, 4});
	Data<Dtype>* data2 = new Data<Dtype>("data2");
	data2->reshape({1, 3, 4, 4});

	param_filler<Dtype> weight_filler(ParamFillerType::Xavier, 0.1);
	weight_filler.fill(data1);
	data1->print_data("data1:");
	for(uint32_t i = 0; i < 1000000000; i++);
	weight_filler.fill(data2);
	data2->print_data("data2:");

	data1->_data->sub_device_mem(data2->_data->device_mem());
	//data1->_data.sub_host_mem(data2->_data.host_mem());
	data1->print_data("data1:");

	delete data1;
	delete data2;

	off();
}

template <typename Dtype>
void ArtisticStyle<Dtype>::flattenTest() {
	on();

	param_filler<Dtype> weight_filler(ParamFillerType::Xavier, 0.1);

	Data<Dtype>* xresp = new Data<Dtype>("xresp");
	xresp->reshape({1, 3, 2, 3});
	weight_filler.fill(xresp);
	xresp->print_data("xresp:");

	SyncMem<Dtype>* flatten = new SyncMem<Dtype>();
	flattenData(xresp, 3, 6, flatten);

	Data<Dtype>* flattend = new Data<Dtype>("flatten");
	flattend->reshape({1, 1, 3, 6});
	flattend->set_host_data(flatten->host_mem());
	flattend->print_data("flattenData:");

	delete xresp;
	delete flatten;
	delete flattend;

	off();
}

template <typename Dtype>
void ArtisticStyle<Dtype>::unflattenTest() {
	on();

	param_filler<Dtype> weight_filler(ParamFillerType::Xavier, 0.1);

	Data<Dtype>* src = new Data<Dtype>("src");
	src->reshape({1, 1, 3, 6});
	weight_filler.fill(src);
	src->print_data("src:");

	Data<Dtype>* dst = new Data<Dtype>("dst");
	dst->reshape({1, 3, 2, 3});

	unflattenData(src->_data.get(), 3, 6, dst);
	dst->print_data("dst:");

	delete src;
	delete dst;

	off();
}

template <typename Dtype>
void ArtisticStyle<Dtype>::gemmTest() {
	on();

	const uint32_t nl = 5;
	const uint32_t height = 3;
	const uint32_t width = 3;
	const uint32_t ml = height*width;

	param_filler<Dtype> weight_filler(ParamFillerType::Xavier, 0.1);

	// nl x nl의 x response
	Data<Dtype>* xresp = new Data<Dtype>("xresp");
	xresp->reshape({1, 1, nl, nl});
	weight_filler.fill(xresp);
	xresp->print_data("xresp:");
	//SyncMem<Dtype>* xresp = new SyncMem<Dtype>();
	//xresp->shape(nl*nl);

	// nl x ml의 feature map
	Data<Dtype>* f = new Data<Dtype>("f");
	f->reshape({1, nl, height, width});
	weight_filler.fill(f);
	f->print_data("f:");

	SyncMem<Dtype>* ff = new SyncMem<Dtype>();
	flattenData(f, nl, ml, ff);


	SyncMem<Dtype>* result = new SyncMem<Dtype>();
	result->reshape(nl*ml);

	const Dtype* xresp_device = xresp->device_data();
	const Dtype* f_device = ff->device_mem();
	Dtype* result_device = result->mutable_device_mem();

	checkCudaErrors(cublasSgemm(Cuda::cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
		nl, ml, nl,
		&Cuda::alpha, xresp_device, nl, f_device, nl,
		&Cuda::beta, result_device, nl));



	Data<Dtype>* resultd = new Data<Dtype>("result");
	resultd->reshape({1, nl, height, width});

	unflattenData(result, nl, ml, resultd);
	resultd->print_data("final result");


	delete xresp;
	delete f;
	delete ff;
	delete result;
	delete resultd;


	off();
}


template <typename Dtype>
void ArtisticStyle<Dtype>::test() {
	cout << "ArtisticStyle test ... " << endl;

	gramMatrixTest();
	//dataSubTest();
	//flattenTest();
	//unflattenTest();
	//gemmTest();
}


template <typename Dtype>
void ArtisticStyle<Dtype>::printCImg(CImg<Dtype>* cimg, const string& head,
		const bool printData) {
	const int spectrum = cimg->spectrum();
	const int height = cimg->height();
	const int width = cimg->width();

	cout << head << ": " << spectrum << "x" << height << "x" << width << endl;

	if (printData) {
		Dtype *data_ptr = cimg->data();
		for (int c = 0; c < spectrum; c++) {
			for (int h = 0; h < height; h++) {
				for (int w = 0; w < width; w++) {
					cout << data_ptr[c*height*width + h*width + w] << ",";
				}
				cout << endl;
			}
			cout << endl << endl;
		}
	}
}




template class ArtisticStyle<float>;

//#endif
