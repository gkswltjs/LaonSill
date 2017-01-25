/*
 * ArtisticStyle.cpp
 *
 *  Created on: 2016. 7. 22.
 *      Author: jhkim
 */


//#ifndef GPU_MODE


#include <stdint.h>
#include <vector>
//#include <CImg.h>

#include "ArtisticStyle.h"
#include "Data.h"
#include "InputLayer.h"

using namespace std;
using namespace cimg_library;


#define CONTENT_
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

//#define STYLE_
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
		  plotContentCost(plotContentCost),
		  plotStyleCost(plotStyleCost) {

	// photo, content image
	prepareCImgFromPath(contentImagePath, this->p);
	updateDimensionFromCImg(this->p);
	updateMean();			// dimension 정보 업데이트 후, 해당 정보 기반  mean값 업데이트.
	preprocess(this->p);
	prepareCImgDisplay(this->p, "content", this->pdisp);

	// art, style image
	prepareCImgFromPath(styleImagePath, this->a);
	preprocess(this->a);
	prepareCImgDisplay(this->a, "style", this->adisp);
	assert(isValidImageDimension(this->a));

	// SyncMem 타입의 xdataTemp와 CImg 타입의 xTemp가 이미지용 메모리 공유--------------------------------------
	this->xdataTemp.reshape(this->channel * this->height * this->width);
	this->xTemp = new CImg<Dtype>(xdataTemp.mutable_host_mem(), this->width, this->height,
			1, this->channel, true);
	//------------------------------------------------------------------------------------

	// input image
	this->xdata = new Data<Dtype>("xdata");
	this->xdata->reshape({1, this->channel, this->height, this->width});
	this->x = new CImg<Dtype>(this->xdata->mutable_host_data(), this->width, this->height,
			1, this->channel, true);
	// xdata를 photo(content)로 초기화
	this->xdata->set_host_data(this->p->data());
	prepareCImgDisplay(this->x, "input", this->xdisp);

	// 이미 preprocess된 this->p를 복사하므로 다시 별도로 x에 대해 preprocess하지 않아야 한다.
	//preprocess(this->x);
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
	//printCImg(cimg);
	normalizeCImg(cimg);
	//printCImg(cimg);
}

template <typename Dtype>
void ArtisticStyle<Dtype>::normalizeCImg(CImg<Dtype>* cimg) {
	Dtype *data_ptr = cimg->data();

	const size_t size = cimg->height() * cimg->width() * cimg->spectrum();
	for (size_t i = 0; i < size; i++) {
		data_ptr[i] /= 255.0f;
	}
}

template <typename Dtype>
void ArtisticStyle<Dtype>::updateMean() {
	this->mean.reshape(this->channel * this->height * this->width);
	Dtype* mean_host = this->mean.mutable_host_mem();
	const uint32_t hxw = this->height * this->width;
	for (uint32_t c = 0; c < this->channel; c++) {
		Dtype value = 0.0;
		if(c == 0) value = 0.47684615850;
		else if(c == 1) value = 0.45469805598;
		else if(c == 2) value = 0.41394191980;
		// RGB 채널 분리되어 있음
		for(uint32_t i = 0; i < hxw; i++) {
			mean_host[i+c*hxw] = value;
		}
	}
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

	if (this->xdata) delete this->xdata;
	//delete this->pdata;
	//delete this->adata;

	if (this->xdisp) delete this->xdisp;
	if (this->pdisp) delete this->pdisp;
	if (this->adisp) delete this->adisp;

	for (uint32_t i = 0; i < this->contentRepLayerResps.size(); i++)
		delete this->contentRepLayerResps[i];
	this->contentRepLayerResps.clear();

	for (uint32_t i = 0; i < this->styleRepLayerResps.size(); i++)
		delete this->styleRepLayerResps[i];
	this->styleRepLayerResps.clear();
}



template <typename Dtype>
void ArtisticStyle<Dtype>::style() {
	// input x 업데이트 횟수
	uint32_t updateCnt = 0;

#ifdef CONTENT_
	// content loss를 계산할 레이어에서 photo에 대한 response를 계산, fixed
	computeContentRepresentationLayerResponses();
#endif
#ifdef STYLE_
	// style loss를 계산할 레이어들에서 art에 대한 response를 계산, fixed
	computeStyleRepresentationLayerResponses();
#endif

	LayersConfig<Dtype>* layersConfig = this->network->getLayersConfig();
	InputLayer<Dtype>* inputLayer = layersConfig->_inputLayer;

	uint32_t i = 0;
	while(true) {
		double contentCost = 0.0;
		double styleCost = 0.0;

		// xdata: input noise x의 Data 타입 객체
		feedforwardWithData(this->xdata);
		for (int i = layersConfig->_layers.size()-1; i >= 0; i--) {
			Layer<Dtype>* repLayer = layersConfig->_layers[i];
			int repLayerIndex = -1;
			if((repLayerIndex = findContentRepLayer(repLayer->getName())) >= 0)
				contentCost += computeContentLossGradientAt(repLayerIndex);
			else if((repLayerIndex = findStyleRepLayer(repLayer->getName())) >= 0)
				styleCost += computeStyleLossGradientAt(repLayerIndex);

			if (repLayerIndex >= 0) {
				for (int j = i; j >= 0; j--) {
					HiddenLayer<Dtype>* hiddenLayer =
							dynamic_cast<HiddenLayer<Dtype>*>(layersConfig->_layers[j]);
					if (hiddenLayer)
						hiddenLayer->backpropagation();
				}
			}
		}

		if (plotContentCost)
			contentCostLogger.addStat(0, "content_cost", contentCost);
		if (plotStyleCost)
			styleCostLogger.addStat(0, "style_cost", styleCost);

		const Dtype* grad_device = inputLayer->_outputData[0]->device_grad();
		Dtype* xdata_device = xdata->mutable_device_data();

		checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(xdata->getCount()),
                                    &learningRate, grad_device, 1, xdata_device, 1));

		if(updateCnt++ % 10 == 0) {
			// device에서 계산된 결과를 host로 업데이트
			// 정식 업데이트 api를 추가해야.
			//xdata->host_data();
			//xdata->_data.print("xdata:");

			//contentLoss->set_mem(xresp->device_mem(), SyncMemCopyType::DeviceToDevice, 
            //                     0, xresp->getSize());
			// xdataTemp device mem을 update한 후,
			xdataTemp.set_mem(xdata->device_data(), SyncMemCopyType::DeviceToDevice, 0,
                              xdata->getCount());
			//xdataTemp.host_mem();
			//xdataTemp.print("xdataTemp before:");
			//xTempDisp->resize(*xTemp, true).display(*xTemp);

			deprocess(&xdataTemp);
			// xdataTemp의 host mem으로 copy해서 가져온다.
			xdataTemp.host_mem();
			//xdataTemp.print("xdataTemp after:");
			xdisp->resize(*xTemp, true).display(*xTemp);


			//CImg<DATATYPE> temp_src(*x);
			//deprocess(&temp_src);
			//xdisp->resize(temp_src, true).display(temp_src.normalize(0, 255));
			//xdisp->resize(temp_src, true).display(temp_src);
			//cout << "reconstruction ... " << i << endl;
		}
		i++;
	}

	while(!xdisp->is_closed()) { xdisp->wait(); }
	while(!pdisp->is_closed()) { pdisp->wait(); }
}


template <typename Dtype>
void ArtisticStyle<Dtype>::computeContentRepresentationLayerResponses() {
	// p 이미지의 buffer를 Data타입의 객체에 복사,
	// feedforward()를 진행
	Data<Dtype>* pdata = new Data<Dtype>("pdata");
	createDataFromCImg(this->p, pdata);
	feedforwardWithData(pdata);
	delete pdata;

	// feedforward() 결과, 지정된 레이어의 response를 복사, 보관 (Content Loss계산용)
	const uint32_t numContentRepLayers = contentRepLayers.size();
	this->contentRepLayerResps.clear();
	this->contentRepLayerResps.resize(numContentRepLayers);
	for(uint32_t i = 0; i < numContentRepLayers; i++) {
		Layer<Dtype>* contentRepLayer = findRepresentationLayer(contentRepLayers[i]);
		//on();
		//contentRepLayer->_outputData[0]->print_data({}, false);
		//off();

		// XXX: 참고한 구현에서 activation된 결과가 아닌 convolution 결과를 사용하고 있어서 수정.
		//SyncMem<Dtype>* contentRepLayerResp =
        //    new SyncMem<Dtype>(contentRepLayer->_outputData[0]->_data.get());

		assert(contentRepLayer);
		ConvLayer<Dtype>* convLayer = dynamic_cast<ConvLayer<Dtype>*>(contentRepLayer);
		assert(convLayer);
		SyncMem<Dtype>* contentRepLayerResp = 
		      new SyncMem<Dtype>(convLayer->_preActivation->_data.get());
		//on();
		//contentRepLayerResp->print("contentRepLayerResp",
		//		contentRepLayer->_outputData[0]->getShape(), false);
		//off();
		this->contentRepLayerResps[i] = contentRepLayerResp;
	}
}

template <typename Dtype>
void ArtisticStyle<Dtype>::computeStyleRepresentationLayerResponses() {
	// a 이미지의 buffer를 Data타입의 객체에 복사,
	// feedforward()를 진행
	Data<Dtype>* adata = new Data<Dtype>("adata");
	createDataFromCImg(a, adata);
	feedforwardWithData(adata);
	delete adata;

	// feedforward() 결과, 지정된 레이어의 response를 복사, 보관 (Style Loss계산용)
	const uint32_t numStyleRepLayers = styleRepLayers.size();
	this->styleRepLayerResps.clear();
	this->styleRepLayerResps.resize(numStyleRepLayers);
	for(uint32_t i = 0; i < numStyleRepLayers; i++) {
		Layer<Dtype>* styleRepLayer = findRepresentationLayer(styleRepLayers[i]);
		SyncMem<Dtype>* styleRepLayerResp = new SyncMem<Dtype>();


		// XXX: 참고한 구현에서 activation된 결과가 아닌 convolution 결과를 사용하고 있어서 수정.
        //createGramMatrixFromData(styleRepLayer->_outputData[0], styleRepLayerResp);
		assert(styleRepLayer);
		ConvLayer<Dtype>* convLayer = dynamic_cast<ConvLayer<Dtype>*>(styleRepLayer);
		assert(convLayer);
		createGramMatrixFromData(convLayer->_preActivation, styleRepLayerResp);

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
	inputLayer->_outputData[0]->set_host_data(data);
	//on();
	//inputLayer->_outputData[0]->print_data({}, false);
	//off();

	for (uint32_t i = 0; i < layersConfig->_layers.size(); i++) {
		//cout << layersConfig->_layers[i]->name << ": feedforward ... " << endl;
		layersConfig->_layers[i]->feedforward();
		if (layersConfig->_layers[i]->name == this->end)
			break;
	}
}



template <typename Dtype>
double ArtisticStyle<Dtype>::computeContentLossGradientAt(const int contentLayerIndex) {
	const string& contentRepLayerName = this->contentRepLayers[contentLayerIndex];

	// index에 해당하는 content representation layer
	HiddenLayer<Dtype>* contentRepLayer = findRepresentationLayer(contentRepLayerName);

	SyncMem<Dtype>* presp = this->contentRepLayerResps[contentLayerIndex];
	SyncMem<Dtype>* xresp = contentRepLayer->_outputData[0]->_data.get();

	// add하는 케이스 때문에 임시 SyncMem 객체를 사용
	SyncMem<Dtype>* contentLoss = new SyncMem<Dtype>();
	contentLoss->reshape(contentRepLayer->_outputData[0]->_grad->getSize());

	// Fl - Pl (Flij < 0 인 케이스 고려하지 않음)
	contentLoss->set_mem(xresp->device_mem(), SyncMemCopyType::DeviceToDevice, 0,
                         xresp->getSize());
	contentLoss->sub_device_mem(presp->device_mem());


	if(this->contentReconstructionFactor != 1.0f)
		contentLoss->scale_device_mem(this->contentReconstructionFactor);

	// content loss를 계산하고 contentRepLayer에 넘겨 backpropagation ...
	// shared input이 아니어서 강제로 contentLoss를 contentRepLayer의 output grad에 copy한다.
	if(contentRepLayerName == this->end)
		contentRepLayer->_outputData[0]->_grad->set_mem(contentLoss->device_mem(),
                                                SyncMemCopyType::DeviceToDevice, 0,
                                                contentLoss->getSize());
	else
		contentRepLayer->_outputData[0]->_grad->add_device_mem(contentLoss->device_mem());

	double contentCost = 0.5*contentLoss->sumsq_device_mem();
	delete contentLoss;

	return contentCost;
}

template <typename Dtype>
double ArtisticStyle<Dtype>::computeStyleLossGradientAt(const int styleLayerIndex) {
	const string& styleRepLayerName = this->styleRepLayers[styleLayerIndex];

	// 1. Style Loss를 계산할 레이어 찾기.
	HiddenLayer<Dtype>* styleRepLayer = findRepresentationLayer(styleRepLayerName);

	// 2. Loss Gradient계산에 필요한 상수 nl과 ml
	const uint32_t nl = styleRepLayer->_outputData[0]->channels();
    // number of feature map elements
	const uint32_t ml = styleRepLayer->_outputData[0]->height() *
			styleRepLayer->_outputData[0]->width();

	// 3. 해당 레이어에 대해 미리 계산해둔 Style Image (Art Image)에 대한 Response 데이터
	SyncMem<Dtype>* aresp = this->styleRepLayerResps[styleLayerIndex];

	// 4. 해당 레이어에 대해 Input Image에 대한 Response 데이터 계산
	SyncMem<Dtype>* xresp = new SyncMem<Dtype>();
	createGramMatrixFromData(styleRepLayer->_outputData[0], xresp);

	// 5. 해당 레이어에 대한 Input Image의 Output Data
	SyncMem<Dtype>* f = styleRepLayer->_outputData[0]->_data.get();
	SyncMem<Dtype>* ff = new SyncMem<Dtype>();
	flattenData(styleRepLayer->_outputData[0], nl, ml, ff);

	// 6. Style Loss를 저장할 SyncMem 객체 생성 (레이어의 Output Grad와 동일한 shape를 가짐)
	//SyncMem<Dtype>* styleLoss = new SyncMem<Dtype>();
	//styleLoss->shape(styleRepLayer->_output->_grad.getSize());

	SyncMem<Dtype>* gemmResult = new SyncMem<Dtype>();
	gemmResult->reshape(nl*ml);

	// 7. Gl = Gl - Al
	xresp->sub_device_mem(aresp->device_mem());

	// 8. Gl = trans(trans(Fl)*Gl)
	const Dtype* xresp_device = xresp->device_mem();
	const Dtype* f_device = ff->device_mem();
	Dtype* gemmResult_device = gemmResult->mutable_device_mem();

	checkCudaErrors(cublasSgemm(Cuda::cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
			nl, ml, nl,
			&Cuda::alpha, xresp_device, nl, f_device, nl,
			&Cuda::beta, gemmResult_device, nl));

	// 9. Gl = 1/(nl*nl*ml*ml)*Gl
	Data<Dtype>* gemmResultd = new Data<Dtype>("gemmResult",
			styleRepLayer->_outputData[0]->getShape());
	unflattenData(gemmResult, nl, ml, gemmResultd);

	gemmResultd->scale_device_data(this->styleReconstructionFactor);

	if(styleRepLayerName == this->end)
		styleRepLayer->_outputData[0]->_grad->set_mem(gemmResultd->_data->device_mem(),
                                              SyncMemCopyType::DeviceToDevice, 0,
                                              gemmResultd->_data->getSize());
	else
		styleRepLayer->_outputData[0]->_grad->add_device_mem(gemmResultd->_data->device_mem());

	delete gemmResult;
	delete gemmResultd;
	delete xresp;
	delete ff;

	double styleCost = xresp->sumsq_device_mem();
	styleCost = 0.25 / (nl * nl * ml * ml) * styleCost * styleCost;

	return styleCost;
}

template <typename Dtype>
void ArtisticStyle<Dtype>::createDataFromCImg(CImg<Dtype>* cimg, Data<Dtype>* data) {
	data->reshape({1, (uint32_t)cimg->spectrum(), (uint32_t)cimg->height(),
		(uint32_t)cimg->width()});
	data->set_device_with_host_data(cimg->data(), 0, cimg->size());

	on();
	data->print_data({}, false);
	off();
}


template <typename Dtype>
void ArtisticStyle<Dtype>::createGramMatrixFromData(Data<Dtype>* data,
		SyncMem<Dtype>* gramMatrix) {
	const uint32_t nl = data->channels();					// number of feature maps
	const uint32_t ml = data->height()*data->width();		// number of feature map elements
	//SyncMem<Dtype>* gramMatrix = new SyncMem<Dtype>();
	gramMatrix->reshape(nl*nl);

	SyncMem<Dtype>* fData = new SyncMem<Dtype>();
	flattenData(data, nl, ml, fData);
	const Dtype* fdata_device = fData->device_mem();

	//const Dtype* data_device = data->device_data();
	Dtype* gram_device = gramMatrix->mutable_device_mem();

	checkCudaErrors(cublasSgemm(Cuda::cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
			nl, nl, ml,
			&Cuda::alpha, fdata_device, nl, fdata_device, nl,
			&Cuda::beta, gram_device, nl));

	//gramMatrix->print("gramMatrix:");
	//gramMatrix->print("gramMatrix:", {1, 1, nl, nl});

	delete fData;

	// TODO
	// 1. 아마 gram matrix도 column major order로 저장해야 할 것.
	// 2. symmetric이라 cmo나 rmo나 차이가 없네.
	// 3. symmectic이니까 반만 계산하도록 수정하자.
	/*
	const Dtype* data_device = data->device_data();
	Dtype* gram_host = gramMatrix->mutable_host_mem();
	for(uint32_t i = 0; i < nl; i++) {
		for(uint32_t j = 0; j < nl; j++) {
			checkCudaErrors(cublasSdot(Cuda::cublasHandle, ml, data_device+i*ml, 1, data_device+j*ml, 1, gram_host+i*nl+j));
		}
	}
	*/
}


template <typename Dtype>
HiddenLayer<Dtype>* ArtisticStyle<Dtype>::findRepresentationLayer(const string& layerName) {
	HiddenLayer<Dtype>* repLayer =
			dynamic_cast<HiddenLayer<Dtype>*>(network->findLayer(layerName));
	if(!repLayer) {
		cout << "specified representation layer is not hidden layer ... " << endl;
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
void ArtisticStyle<Dtype>::preprocess(CImg<Dtype>* cimg) {
	const Dtype* mean_ptr = mean.host_mem();
	Dtype *data_ptr = cimg->data();
	size_t size = cimg->height()*cimg->width()*cimg->spectrum();
	for(size_t i = 0; i < size; i++) {
		data_ptr[i] -= mean_ptr[i];
	}
	//printCImg(cimg);
}


template <typename Dtype>
void ArtisticStyle<Dtype>::preprocess(SyncMem<Dtype>* mem) {
	mem->sub_device_mem(mean.device_mem());
}


template <typename Dtype>
void ArtisticStyle<Dtype>::deprocess(CImg<Dtype>* cimg) {
	const Dtype* mean_ptr = mean.host_mem();
	Dtype *data_ptr = cimg->data();
	size_t size = cimg->height()*cimg->width()*cimg->spectrum();
	for(uint32_t i = 0; i < size; i++) {
		data_ptr[i] += mean_ptr[i];
	}

	/*
	Dtype* data_ptr = cimg->data();
	const int height = cimg->height();
	const int width = cimg->width();
	const int channel = cimg->spectrum();

	for(int c = 0; c < channel; c++) {
		for(int h = 0; h < height; h++) {
			for(int w = 0; w < width; w++) {
				data_ptr[w+h*width+c*height*width] += mean[c];
			}
		}
	}
	*/
}

template <typename Dtype>
void ArtisticStyle<Dtype>::deprocess(SyncMem<Dtype>* mem) {
	mem->add_device_mem(mean.device_mem());
}



template <typename Dtype>
void ArtisticStyle<Dtype>::clipImage(CImg<Dtype>* cimg) {
	/*
	Dtype *src_ptr = cimg->data();
	const int width = cimg->width();
	const int height = cimg->height();
	const int channel = cimg->spectrum();
	int index;
	for(int c = 0; c < channel; c++) {
		for(int h = 0; h < height; h++) {
			for(int w = 0; w < width; w++) {
				index = w+h*width+c*width*height;
				if(src_ptr[index] < -mean[c]) {
					src_ptr[index] = -mean[c];
				} else if(src_ptr[index] > 1.0-mean[c]) {
					src_ptr[index] = 1.0-mean[c];
				}
			}
		}
	}
	*/
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

	on();
	data->print_data("data:");
	off();


	SyncMem<Dtype>* gramMatrix = new SyncMem<Dtype>();
	createGramMatrixFromData(data, gramMatrix);
	gramMatrix->print("gramMatrix:");
	gramMatrix->print("gramMatrix:", {1, 1, numFeatureMaps, numFeatureMaps});
	delete gramMatrix;
}

template <typename Dtype>
void ArtisticStyle<Dtype>::dataSubTest() {
	Data<Dtype>::printConfig = 1;

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

	Data<Dtype>::printConfig = 0;
}

template <typename Dtype>
void ArtisticStyle<Dtype>::flattenTest() {
	Data<Dtype>::printConfig = 1;

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

	Data<Dtype>::printConfig = 0;
}

template <typename Dtype>
void ArtisticStyle<Dtype>::unflattenTest() {
	Data<Dtype>::printConfig = 1;

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

	Data<Dtype>::printConfig = 0;
}

template <typename Dtype>
void ArtisticStyle<Dtype>::gemmTest() {
	Data<Dtype>::printConfig = 1;

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


	Data<Dtype>::printConfig = 0;
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
void ArtisticStyle<Dtype>::printCImg(CImg<Dtype>* cimg) {
	Dtype *data_ptr = cimg->data();

	const int spectrum = cimg->spectrum();
	const int height = cimg->height();
	const int width = cimg->width();

	cout << spectrum << "x" << height << "x" << width << endl;
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



template class ArtisticStyle<float>;

//#endif
