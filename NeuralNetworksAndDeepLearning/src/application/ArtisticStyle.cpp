/*
 * ArtisticStyle.cpp
 *
 *  Created on: 2016. 7. 22.
 *      Author: jhkim
 */

#include "ArtisticStyle.h"

//#ifndef GPU_MODE


#include <stdint.h>
#include <vector>
//#include <CImg.h>

#include "../Data.h"
#include "../layer/InputLayer.h"

using namespace std;
using namespace cimg_library;


#define CONTENT_
#define STYLE_



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
	this->p = new CImg<Dtype>(contentImagePath.c_str());
	this->p->normalize(0.0f, 1.0f);


	this->width = this->p->width();
	this->height = this->p->height();
	this->channel = this->p->spectrum();

	mean.shape(channel*height*width);
	Dtype* mean_host = mean.mutable_host_mem();
	const uint32_t hxw = height*width;
	for(uint32_t c = 0; c < channel; c++) {
		Dtype value = 0.0;
		if(c == 0) value = 0.47684615850;
		else if(c == 1) value = 0.45469805598;
		else if(c == 2) value = 0.41394191980;
		for(uint32_t i = 0; i < hxw; i++) {
			mean_host[i+c*hxw] = value;
		}
	}

	xdataTemp.shape(channel*height*width);
	xTemp = new CImg<Dtype>(xdataTemp.mutable_host_mem(), width, height, 1, channel, true);
	//this->xTempDisp = new CImgDisplay(*xTemp, "xTemp");

	preprocess(this->p);
	this->pdisp = new CImgDisplay(*p, "content");



	this->contentGrad.shape(this->network->getLayersConfig()->_inputLayer->_output->getCount());

	// art, style image
	this->a = new CImg<Dtype>(styleImagePath.c_str());
	this->a->normalize(0.0f, 1.0f);
	preprocess(this->a);
	this->adisp = new CImgDisplay(*a, "style");

	if(this->width != a->width() ||
			this->height != a->height() ||
			this->channel != a->spectrum()) {
		cout << "p and a image dimensions are not identical ... " << endl;
		exit(-1);
	}
	this->styleGrad.shape(this->network->getLayersConfig()->_inputLayer->_output->getCount());



	// input image
	this->xdata = new Data<Dtype>();
	this->xdata->shape({1, channel, height, width});

	this->x = new CImg<Dtype>(this->xdata->mutable_host_data(), width, height, 1, channel, true);
	this->xdata->set_host_data(this->p->data());
	//this->x->noise(0.1);
	//this->x->noise(10);
	//this->x->normalize(0.0f, 1.0f);
	this->xdisp = new CImgDisplay(*x, "input");
	preprocess(this->x);

}




template <typename Dtype>
ArtisticStyle<Dtype>::~ArtisticStyle() {
	if(network) delete network;

	if(x) delete x;
	if(p) delete p;
	if(a) delete a;

	if(xdata) delete xdata;
	//delete pdata;
	//delete adata;

	if(xdisp) delete xdisp;
	if(pdisp) delete pdisp;
	if(adisp) delete adisp;

	for(uint32_t i = 0; i < contentRepLayerResps.size(); i++) delete contentRepLayerResps[i];
	contentRepLayerResps.clear();

	for(uint32_t i = 0; i < styleRepLayerResps.size(); i++) delete styleRepLayerResps[i];
	styleRepLayerResps.clear();
}



template <typename Dtype>
void ArtisticStyle<Dtype>::style() {

	uint32_t updateCnt = 0;


#ifdef CONTENT_
	// content loss를 계산할 레이어에서 photo에 대한 response를 계산, fixed
	computeContentRepresentationLayerResponses();
#endif
#ifdef STYLE_
	// style loss를 계산할 레이어들에서 art에 대한 response를 계산, fixed
	computeStyleRepresentationLayerResponses();
#endif

	InputLayer<Dtype>* inputLayer = network->getLayersConfig()->_inputLayer;
	uint32_t i = 0;
	while(true) {
	//for(i = 0; i < 1000; ) {
		// 현재 input에 대해 feedforward하여
		// content/ style 각각의 representation 레이어에서의 loss 계산,
		// input gradient를 계산
		//on();
		//xdata->print_data("xdata:");
		//off();
		inputLayer->feedforward(0, xdata, end.c_str());

		HiddenLayer<Dtype>* repLayer = findRepresentationLayer(end);
		double contentCost = 0.0;
		double styleCost = 0.0;
		while(repLayer) {
			//cout << "compute x representation at layer " << repLayer->getName() << endl;

			int repLayerIndex = -1;
			if((repLayerIndex = findContentRepLayer(repLayer->getName())) >= 0) {
				//cout << "found representation layer in content representation layers ... " << endl;
				contentCost += computeContentLossGradientAt(repLayerIndex);
			} else if((repLayerIndex = findStyleRepLayer(repLayer->getName())) >= 0) {
				//cout << "found representation layer in style representation layers ... " << endl;
				styleCost += computeStyleLossGradientAt(repLayerIndex);
			}
			//on();
			//repLayer->_output->print_grad("output_grad:");
			//off();
			repLayer->_backpropagation();
			//on();
			//repLayer->_input->print_grad("input_grad:");
			//off();

			repLayer = dynamic_cast<HiddenLayer<Dtype>*>(repLayer->getPrevLayers()[0]);
		}
		if(plotContentCost) contentCostLogger.addStat(0, "content_cost", contentCost);
		if(plotStyleCost) styleCostLogger.addStat(0, "style_cost", styleCost);
		//cout << "contentCost: " << contentCost << endl;

		//styleGrad.add_device_mem(network->config->_inputLayer->getNextLayers()[0]->_input->_grad.device_mem());
		const Dtype* grad_device = inputLayer->getNextLayers()[0]->_input->_grad.device_mem();
		Dtype* xdata_device = xdata->mutable_device_data();


		//on();
		//inputLayer->getNextLayers()[0]->_input->print_grad("inputGrad:");
		//xdata->print_data("inputData:");
		//off();

		checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(xdata->getCount()), &learningRate, grad_device, 1, xdata_device, 1));
		//on();
		//xdata->print_data("inputData:");
		//off();


		if(updateCnt++ % 10 == 0) {
			// device에서 계산된 결과를 host로 업데이트
			// 정식 업데이트 api를 추가해야.
			//xdata->host_data();
			//xdata->_data.print("xdata:");

			//contentLoss->set_mem(xresp->device_mem(), SyncMemCopyType::DeviceToDevice, 0, xresp->getSize());
			// xdataTemp device mem을 update한 후,
			xdataTemp.set_mem(xdata->device_data(), SyncMemCopyType::DeviceToDevice, 0, xdata->getCount());
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

		//Util::printVramInfo();
	}

	while(!xdisp->is_closed()) { xdisp->wait(); }
	while(!pdisp->is_closed()) { pdisp->wait(); }
}


template <typename Dtype>
void ArtisticStyle<Dtype>::computeContentRepresentationLayerResponses() {
	// p 이미지의 buffer를 Data타입의 객체에 복사,
	// feedforward()를 진행
	Data<Dtype>* pdata = createDataFromCImg(p);
	network->getLayersConfig()->_inputLayer->feedforward(0, pdata, end.c_str());

	// feedforward() 결과, 지정된 레이어의 response를 복사, 보관 (Content Loss계산용)
	const uint32_t numContentRepLayers = contentRepLayers.size();
	for(uint32_t i = 0; i < numContentRepLayers; i++) {
		//Layer<Dtype>* contentRepLayer = network->findLayer(contentRepLayers[i]);
		Layer<Dtype>* contentRepLayer = findRepresentationLayer(contentRepLayers[i]);
		SyncMem<Dtype>* contentRepLayerResp = new SyncMem<Dtype>(contentRepLayer->_output->_data);
		contentRepLayerResps.push_back(contentRepLayerResp);
	}
	delete pdata;
}

template <typename Dtype>
void ArtisticStyle<Dtype>::computeStyleRepresentationLayerResponses() {
	// a 이미지의 buffer를 Data타입의 객체에 복사,
	// feedforward()를 진행
	Data<Dtype>* adata = createDataFromCImg(a);
	//on();
	//adata->print_data("adata:");
	//off();
	network->getLayersConfig()->_inputLayer->feedforward(0, adata, end.c_str());

	// feedforward() 결과, 지정된 레이어의 response를 복사, 보관 (Style Loss계산용)
	const uint32_t numStyleRepLayers = styleRepLayers.size();
	for(uint32_t i = 0; i < numStyleRepLayers; i++) {
		//Layer<Dtype>* styleRepLayer = network->findLayer(styleRepLayers[i]);
		Layer<Dtype>* styleRepLayer = findRepresentationLayer(styleRepLayers[i]);
		//cout << "compute style representation at layer " << styleRepLayer->getName() << endl;

		SyncMem<Dtype>* styleRepLayerResp = createGramMatrixFromData(styleRepLayer->_output.get());

		// layer feature map으로부터 gram matrix가 제대로 생성되는지 확인할 것.
		//on();
		//styleRepLayer->_output->print_data(styleRepLayer->getName());
		//styleRepLayerResp->print("styleRepLayerResp:");
		//off();
		// 대상 레이어 index에 맞춰 gram matrix 저장
		styleRepLayerResps.push_back(styleRepLayerResp);
		//cout << "push back representation to vector at " << i << endl;
	}
	delete adata;
}



template <typename Dtype>
double ArtisticStyle<Dtype>::computeContentLossGradientAt(const int contentLayerIndex) {
	const string& contentRepLayerName = contentRepLayers[contentLayerIndex];

	// index에 해당하는 content representation layer
	HiddenLayer<Dtype>* contentRepLayer = findRepresentationLayer(contentRepLayerName);

	SyncMem<Dtype>* presp = contentRepLayerResps[contentLayerIndex];
	SyncMem<Dtype>* xresp = &contentRepLayer->_output->_data;
	SyncMem<Dtype>* contentLoss = new SyncMem<Dtype>();
	contentLoss->shape(contentRepLayer->_output->_grad.getSize());
	//SyncMem<Dtype>* cost = new SyncMem<Dtype>();
	//cost->shape(contentRepLayer->_output->_grad.getSize());


	contentLoss->set_mem(xresp->device_mem(), SyncMemCopyType::DeviceToDevice, 0, xresp->getSize());
	contentLoss->sub_device_mem(presp->device_mem());


	/*
	const Dtype* presp_host = presp->host_mem();
	const Dtype* xresp_host = xresp->host_mem();
	Dtype* contentLoss_host = contentLoss->mutable_host_mem();
	Dtype* cost_host = cost->mutable_host_mem();

	const uint32_t size = contentLoss->getSize();
	for(uint32_t i = 0; i < size; i++) {
		cost_host[i] = xresp_host[i] - presp_host[i];
		//if(xresp_host[i] > 0) contentLoss_host[i] = cost_host[i];
		//else contentLoss_host[i] = 0.0;
		contentLoss_host[i] = cost_host[i];
		//if(xresp_host[i] > 0) contentLoss_host[i] = xresp_host[i] - presp_host[i];
		//else contentLoss_host[i] = 0.0;
	}
	*/
	double contentCost = 0.5*contentLoss->sumsq_device_mem();
	//costLogger.addStat(0, "content_cost:", contentCost);


	//double norm = contentLoss->asum_device_mem() + 0.00000001;		//1e-8
	//double weight = 0.02 / norm;
	//double weight = 1;
	if(contentReconstructionFactor != 1) {
		contentLoss->scale_device_mem(contentReconstructionFactor);
	}
	//cout << "content reconstruction norm: " << norm << ", weight: " << weight << endl;

	//contentLoss->print("contentLoss: ");
	// content loss를 계산하고 contentRepLayer에 넘겨 backpropagation ...
	// shared input이 아니어서 강제로 contentLoss를 contentRepLayer의 output grad에 copy해줘야 한다.
	if(contentRepLayerName == end) {
		contentRepLayer->_output->_grad.set_mem(contentLoss->device_mem(), SyncMemCopyType::DeviceToDevice, 0, contentLoss->getSize());
	} else {
		contentRepLayer->_output->_grad.add_device_mem(contentLoss->device_mem());
	}

	//contentRepLayer->_output->_grad.print("outputGrad:");
	//contentRepLayer->backpropagation(0, 0, 0);
	//contentRepLayer->_backpropagation();

	delete contentLoss;
	//delete cost;
	return contentCost;
}

template <typename Dtype>
double ArtisticStyle<Dtype>::computeStyleLossGradientAt(const int styleLayerIndex) {
	const string& styleRepLayerName = styleRepLayers[styleLayerIndex];

	// 1. Style Loss를 계산할 레이어 찾기.
	HiddenLayer<Dtype>* styleRepLayer = findRepresentationLayer(styleRepLayerName);
	//cout << "style representation layer " << styleRepLayer->getName() << " of index " << styleLayerIndex << endl;

	// 6. Loss Gradient계산에 필요한 상수 nl과 ml
	const uint32_t nl = styleRepLayer->_output->channels();
	const uint32_t ml = styleRepLayer->_output->height()*styleRepLayer->_output->width();		// number of feature map elements

	// 2. 해당 레이어에 대해 미리 계산해둔 Style Image (Art Image)에 대한 Response 데이터
	SyncMem<Dtype>* aresp = styleRepLayerResps[styleLayerIndex];
	//aresp->print("aresp:");

	// 3. 해당 레이어에 대해 Input Image에 대한 Response 데이터 계산
	//on();
	//styleRepLayer->_output->print_data("styleRepLayer_output:");
	//off();
	SyncMem<Dtype>* xresp = createGramMatrixFromData(styleRepLayer->_output.get());
	//xresp->print("xresp:");

	// 4. 해당 레이어에 대한 Input Image의 Output Data
	SyncMem<Dtype>* f = &styleRepLayer->_output->_data;
	//f->print("f:");
	SyncMem<Dtype>* ff = flattenData(styleRepLayer->_output.get(), nl, ml);
	//ff->print("ff:");
	//on();
	//styleRepLayer->_output->print_data("styleRepLayer_output:");
	//off();

	// 5. Style Loss를 저장할 SyncMem 객체 생성 (해당 레이어의 Output Grad와 동일한 shape를 가짐)
	//SyncMem<Dtype>* styleLoss = new SyncMem<Dtype>();
	//styleLoss->shape(styleRepLayer->_output->_grad.getSize());

	SyncMem<Dtype>* gemmResult = new SyncMem<Dtype>();
	gemmResult->shape(nl*ml);

	// 7. Gl = Gl - Al
	xresp->sub_device_mem(aresp->device_mem());

	double styleCost = xresp->sumsq_device_mem();
	styleCost = 0.25/nl/nl/ml/ml*styleCost*styleCost;



	//xresp->print("G-A:");

	// 8. Gl = trans(trans(Fl)*Gl)
	const Dtype* xresp_device = xresp->device_mem();
	const Dtype* f_device = ff->device_mem();
	Dtype* gemmResult_device = gemmResult->mutable_device_mem();

	checkCudaErrors(cublasSgemm(Cuda::cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
			nl, ml, nl,
			&Cuda::alpha, xresp_device, nl, f_device, nl,
			&Cuda::beta, gemmResult_device, nl));

	//gemmResult->print("gemm result:");

	// 9. Gl = 1/(nl*nl*ml*ml)*Gl

	//size_t denom = ml;
	//denom *= nl*nl*ml*styleRepLayers.size();
	//double scale = styleReconstructionFactor/denom;
	//cout << "factor: " << styleReconstructionFactor << ", nl: " << nl << ", ml: " << ml << ", denom: " << denom << ", scale: " << scale << endl;
	//gemmResult->scale_device_mem(scale);

	Data<Dtype>* gemmResultd = new Data<Dtype>(styleRepLayer->_output->getShape());
	unflattenData(gemmResult, nl, ml, gemmResultd);
	//on();
	//gemmResultd->print_data("unflattened:");
	//off();


	/*
	const Dtype* result_host = gemmResultd->host_data();
	const Dtype* f_host = f->host_mem();
	Dtype* styleLoss_host = styleLoss->mutable_host_mem();

	const uint32_t size = styleLoss->getSize();
	for(uint32_t i = 0; i < size; i++) {
		//if(f_host[i] >= 0) styleLoss_host[i] = result_host[i];
		//else styleLoss_host[i] = 0;
		styleLoss_host[i] = result_host[i];
	}

	//double norm = styleLoss->asum_device_mem();
	//double norm = sqrt(styleLoss->sumsq_device_mem());
	//double weight = 1.0/norm;
	*/


	//double weight = 0.0001;
	//styleLoss->scale_device_mem(weight);
	gemmResultd->scale_device_data(styleReconstructionFactor);
	//cout << "style reconstruction norm: " << norm << ", weight: " << weight << endl;

	//on();
	//styleRepLayer->_output->print_grad("outputGrad:");
	//off();
	//styleLoss->print("styleLoss:");



	if(styleRepLayerName == end) {
		styleRepLayer->_output->_grad.set_mem(gemmResultd->_data.device_mem(), SyncMemCopyType::DeviceToDevice, 0, gemmResultd->_data.getSize());
	} else {
		styleRepLayer->_output->_grad.add_device_mem(gemmResultd->_data.device_mem());
	}
	//on();
	//styleRepLayer->_output->print_grad("outputGrad:");
	//off();

	//delete styleLoss;
	delete gemmResult;
	delete gemmResultd;
	delete xresp;
	delete ff;

	return styleCost;
}

















































/*

template <typename Dtype>
void ArtisticStyle<Dtype>::computeContentLossGradient() {
	const uint32_t numContentRepLayers = contentRepLayers.size();

	contentGrad.reset_device_mem();
	for(uint32_t i = 0; i < numContentRepLayers; i++) {
		// i번째 content representation layer에서 발생한
		// loss에 대한 gradient를 구해 contentGrad에 누적시킨다.
		computeContentLoss(i);
	}
	if(numContentRepLayers > 1) {
		const float scale = 1.0f/numContentRepLayers;
		contentGrad.scale_device_mem(scale);
	}
}


template <typename Dtype>
void ArtisticStyle<Dtype>::computeStyleLossGradient() {
	const uint32_t numStyleRepLayers = styleRepLayers.size();

	styleGrad.reset_device_mem();
	for(uint32_t i = 0; i < numStyleRepLayers; i++) {
		// i번째 style representation layer에서 발생한
		// loss에 대한 gradient를 구해 styleGrad에 누적시킨다.
		computeStyleLoss(i);
	}
	if(numStyleRepLayers > 1) {
		const float scale = 1.0f/numStyleRepLayers;
		styleGrad.scale_device_mem(scale);
	}
}



template <typename Dtype>
void ArtisticStyle<Dtype>::computeContentLoss(uint32_t contentRepLayerIndex) {
	// index에 해당하는 content representation layer
	HiddenLayer<Dtype>* contentRepLayer = findRepresentationLayer(contentRepLayers[contentRepLayerIndex]);

	SyncMem<Dtype>* presp = contentRepLayerResps[contentRepLayerIndex];
	SyncMem<Dtype>* xresp = &contentRepLayer->_output->_data;
	SyncMem<Dtype>* contentLoss = new SyncMem<Dtype>();
	contentLoss->shape(contentRepLayer->_output->_grad.getSize());

	const Dtype* presp_host = presp->host_mem();
	const Dtype* xresp_host = xresp->host_mem();
	Dtype* contentLoss_host = contentLoss->mutable_host_mem();

	const uint32_t size = contentLoss->getSize();
	for(uint32_t i = 0; i < size; i++) {
		if(xresp_host[i] > 0) contentLoss_host[i] = contentReconstructionFactor*(xresp_host[i] - presp_host[i]);
		else contentLoss_host[i] = 0.0;
	}

	//contentLoss->print("contentLoss: ");
	// content loss를 계산하고 contentRepLayer에 넘겨 backpropagation ...
	// shared input이 아니어서 강제로 contentLoss를 contentRepLayer의 output grad에 copy해줘야 한다.
	contentRepLayer->_output->_grad.set_mem(contentLoss->host_mem(), SyncMemCopyType::HostToDevice, 0, contentLoss->getSize());
	//contentRepLayer->_output->_grad.print("outputGrad:");
	contentRepLayer->backpropagation(0, 0, 0);

	// input에 전달된 gradient를 contentGrad에 합산
	contentGrad.add_device_mem(network->config->_inputLayer->getNextLayers()[0]->_input->_grad.device_mem());

	delete contentLoss;
}






template <typename Dtype>
void ArtisticStyle<Dtype>::computeStyleLoss(uint32_t styleRepLayerIndex) {
	// 1. Style Loss를 계산할 레이어 찾기.
	HiddenLayer<Dtype>* styleRepLayer = findRepresentationLayer(styleRepLayers[styleRepLayerIndex]);

	// 6. Loss Gradient계산에 필요한 상수 nl과 ml
	const uint32_t nl = styleRepLayer->_output->channels();
	const uint32_t ml = styleRepLayer->_output->height()*styleRepLayer->_output->width();		// number of feature map elements



	// 2. 해당 레이어에 대해 미리 계산해둔 Style Image (Art Image)에 대한 Response 데이터
	SyncMem<Dtype>* aresp = styleRepLayerResps[styleRepLayerIndex];
	//aresp->print("aresp:");

	// 3. 해당 레이어에 대해 Input Image에 대한 Response 데이터 계산
	SyncMem<Dtype>* xresp = createGramMatrixFromData(styleRepLayer->_output.get());
	//xresp->print("xresp:");

	// 4. 해당 레이어에 대한 Input Image의 Output Data
	SyncMem<Dtype>* f = &styleRepLayer->_output->_data;
	//on();
	//styleRepLayer->_output->print_data("f:");
	//off();

	SyncMem<Dtype>* ff = flattenData(styleRepLayer->_output.get(), nl, ml);
	//ff->print("ff:");

	// 5. Style Loss를 저장할 SyncMem 객체 생성 (해당 레이어의 Output Grad와 동일한 shape를 가짐)
	SyncMem<Dtype>* styleLoss = new SyncMem<Dtype>();
	styleLoss->shape(styleRepLayer->_output->_grad.getSize());

	SyncMem<Dtype>* gemmResult = new SyncMem<Dtype>();
	gemmResult->shape(nl*ml);


	// 7. Gl = Gl - Al
	xresp->sub_device_mem(aresp->device_mem());
	//xresp->print("G-A:");

	// 8. Gl = trans(trans(Fl)*Gl)
	const Dtype* xresp_device = xresp->device_mem();
	const Dtype* f_device = ff->device_mem();
	Dtype* gemmResult_device = gemmResult->mutable_device_mem();

	checkCudaErrors(cublasSgemm(Cuda::cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
			nl, ml, nl,
			&Cuda::alpha, xresp_device, nl, f_device, nl,
			&Cuda::beta, gemmResult_device, nl));

	//gemmResult->print("gemm result:");

	// 9. Gl = 1/(nl*nl*ml*ml)*Gl

	size_t denom = ml;
	denom *= nl*nl*ml;
	double scale = styleReconstructionFactor/denom;
	cout << "factor: " << styleReconstructionFactor << ", nl: " << nl << ", ml: " << ml << ", denom: " << denom << ", scale: " << scale << endl;
	//gemmResult->scale_device_mem(scale);
	//gemmResult->print("gemm scale result:");
	//gemmResult->scale_device_mem(1.0f/(nl*ml));

	Data<Dtype>* gemmResultd = new Data<Dtype>(styleRepLayer->_output->getShape());
	unflattenData(gemmResult, nl, ml, gemmResultd);
	//on();
	//gemmResultd->print_data("unflattened:");
	//off();


	const Dtype* result_host = gemmResultd->host_data();
	const Dtype* f_host = f->host_mem();
	Dtype* styleLoss_host = styleLoss->mutable_host_mem();

	const uint32_t size = styleLoss->getSize();
	for(uint32_t i = 0; i < size; i++) {
		if(f_host[i] > 0) styleLoss_host[i] = result_host[i];
		else styleLoss_host[i] = 0;
	}

	styleRepLayer->_output->_grad.set_mem(styleLoss->host_mem(), SyncMemCopyType::HostToDevice, 0, styleLoss->getSize());
	//on();
	//styleRepLayer->_output->print_grad("outputGrad:");
	styleRepLayer->backpropagation(0, 0, 0);
	//off();

	//styleGrad.print("styleGrad:");
	// input에 전달된 gradient를 styleGrad에 합산
	styleGrad.add_device_mem(network->config->_inputLayer->getNextLayers()[0]->_input->_grad.device_mem());
	//styleGrad.print("styleGrad:");

	delete styleLoss;
	delete gemmResult;
	delete gemmResultd;
	delete xresp;
	delete ff;
}
*/








template <typename Dtype>
Data<Dtype>* ArtisticStyle<Dtype>::createDataFromCImg(CImg<Dtype>* cimg) {
	Data<Dtype>* data = new Data<Dtype>();
	data->shape({1, (uint32_t)cimg->spectrum(), (uint32_t)cimg->height(), (uint32_t)cimg->width()});
	data->set_device_with_host_data(cimg->data(), 0, cimg->size());
	return data;
}


template <typename Dtype>
SyncMem<Dtype>* ArtisticStyle<Dtype>::createGramMatrixFromData(Data<Dtype>* data) {
	const uint32_t nl = data->channels();					// number of feature maps
	const uint32_t ml = data->height()*data->width();		// number of feature map elements
	SyncMem<Dtype>* gramMatrix = new SyncMem<Dtype>();
	gramMatrix->shape(nl*nl);

	SyncMem<Dtype>* fData = flattenData(data, nl, ml);
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

	return gramMatrix;
}


template <typename Dtype>
HiddenLayer<Dtype>* ArtisticStyle<Dtype>::findRepresentationLayer(const string& layerName) {
	HiddenLayer<Dtype>* repLayer = dynamic_cast<HiddenLayer<Dtype>*>(network->findLayer(layerName));
	if(!repLayer) {
		cout << "specified representation layer is not hidden layer ... " << endl;
		exit(1);
	}
	return repLayer;
}






template <typename Dtype>
SyncMem<Dtype>* ArtisticStyle<Dtype>::flattenData(Data<Dtype>* data, const uint32_t flattenHeight, const uint32_t flattenWidth) {

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

	SyncMem<Dtype>* mem = new SyncMem<Dtype>();
	mem->shape(pfs);

	const Dtype* data_host = data->host_data();
	Dtype* mem_host = mem->mutable_host_mem();

	for(uint32_t w = 0; w < pow; w++) {
		for(uint32_t h = 0; h < poh; h++) {
			for(uint32_t c = 0; c < poc; c++) {
				mem_host[c+(h+w*poh)*pfw] = data_host[c*poHxW+h*pow+w];
			}
		}
	}
	return mem;
}


template <typename Dtype>
void ArtisticStyle<Dtype>::unflattenData(SyncMem<Dtype>* mem, const uint32_t flattenHeight, const uint32_t flattenWidth, Data<Dtype>* data) {
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
	for(uint32_t i = 0; i < size; i++) {
		data_ptr[i] -= mean_ptr[i];
	}
	/*
	for(int c = 0; c < channel; c++) {
		for(int h = 0; h < height; h++) {
			for(int w = 0; w < width; w++) {
				data_ptr[w+h*width+c*height*width] -= mean[c];
			}
		}
	}
	*/
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
	Data<Dtype>* data = new Data<Dtype>();
	data->shape({1, numFeatureMaps, 4, 4});

	param_filler<Dtype> weight_filler(ParamFillerType::Xavier, 0.1);
	weight_filler.fill(data);

	//on();
	//data->print_data("data:");
	//off();


	SyncMem<Dtype>* gramMatrix = createGramMatrixFromData(data);
	gramMatrix->print("gramMatrix:");
	gramMatrix->print("gramMatrix:", {1, 1, numFeatureMaps, numFeatureMaps});
}

template <typename Dtype>
void ArtisticStyle<Dtype>::dataSubTest() {
	Data<Dtype>::printConfig = 1;

	Data<Dtype>* data1 = new Data<Dtype>();
	data1->shape({1, 3, 4, 4});
	Data<Dtype>* data2 = new Data<Dtype>();
	data2->shape({1, 3, 4, 4});

	param_filler<Dtype> weight_filler(ParamFillerType::Xavier, 0.1);
	weight_filler.fill(data1);
	data1->print_data("data1:");
	for(uint32_t i = 0; i < 1000000000; i++);
	weight_filler.fill(data2);
	data2->print_data("data2:");

	data1->_data.sub_device_mem(data2->_data.device_mem());
	//data1->_data.sub_host_mem(data2->_data.host_mem());
	data1->print_data("data1:");

	Data<Dtype>::printConfig = 0;
}

template <typename Dtype>
void ArtisticStyle<Dtype>::flattenTest() {
	Data<Dtype>::printConfig = 1;

	param_filler<Dtype> weight_filler(ParamFillerType::Xavier, 0.1);

	Data<Dtype>* xresp = new Data<Dtype>();
	xresp->shape({1, 3, 2, 3});
	weight_filler.fill(xresp);
	xresp->print_data("xresp:");

	SyncMem<Dtype>* flatten = flattenData(xresp, 3, 6);
	Data<Dtype>* flattend = new Data<Dtype>();
	flattend->shape({1, 1, 3, 6});
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

	Data<Dtype>* src = new Data<Dtype>();
	src->shape({1, 1, 3, 6});
	weight_filler.fill(src);
	src->print_data("src:");

	Data<Dtype>* dst = new Data<Dtype>();
	dst->shape({1, 3, 2, 3});

	unflattenData(&src->_data, 3, 6, dst);
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
	Data<Dtype>* xresp = new Data<Dtype>();
	xresp->shape({1, 1, nl, nl});
	weight_filler.fill(xresp);
	xresp->print_data("xresp:");
	//SyncMem<Dtype>* xresp = new SyncMem<Dtype>();
	//xresp->shape(nl*nl);

	// nl x ml의 feature map
	Data<Dtype>* f = new Data<Dtype>();
	f->shape({1, nl, height, width});
	weight_filler.fill(f);
	f->print_data("f:");

	SyncMem<Dtype>* ff = flattenData(f, nl, ml);


	SyncMem<Dtype>* result = new SyncMem<Dtype>();
	result->shape(nl*ml);

	const Dtype* xresp_device = xresp->device_data();
	const Dtype* f_device = ff->device_mem();
	Dtype* result_device = result->mutable_device_mem();

	checkCudaErrors(cublasSgemm(Cuda::cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
		nl, ml, nl,
		&Cuda::alpha, xresp_device, nl, f_device, nl,
		&Cuda::beta, result_device, nl));



	Data<Dtype>* resultd = new Data<Dtype>();
	resultd->shape({1, nl, height, width});

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














/*
template <typename Dtype>
void readLayerInfo(Network* network, int numLayers, const char **layerName, LayerInfo_t *layerInfos) {

	for(int l = 0; l < numLayers; l++) {
		// find destination layer and first hidden layer ///////
		HiddenLayer* dstLayer = dynamic_cast<HiddenLayer*>(network->findLayer(layerName[l]));
		if(!dstLayer) {
			cout << "could not find layer of name " << layerName[l] << " ... " << endl;
			exit(-1);
		}
		io_dim dstLayerOutDim = dstLayer->getOutDimension();

		layerInfos[l].layer = dstLayer;
		layerInfos[l].out_dim = dstLayerOutDim;
		layerInfos[l].outSize = dstLayerOutDim.unitsize();
		layerInfos[l].N = dstLayerOutDim.channels;
		layerInfos[l].M = dstLayerOutDim.rows*dstLayerOutDim.cols;
	}

}


template <typename Dtype>
void ArtisticStyle::style(const char* content_img_path, const char* style_img_path,
		const char* end) {

	// preparing content image /////////////////////////////
	CImg<DATATYPE> content_img(content_img_path);
	CImgDisplay content_disp(content_img, "content");
	content_img.normalize(0.0f, 1.0f);
	//preprocess(content_img);
	////////////////////////////////////////////////////////////////

#ifdef STYLE_
	// preparing style image ///////////////////////////////
	CImg<DATATYPE> style_img(style_img_path);
	if(content_img.width() != style_img.width() ||
			content_img.height() != style_img.height() ||
			content_img.spectrum() != style_img.spectrum()) {
		cout << "input image dimensions are not identical ... " << endl;
		exit(-1);
	}
	CImgDisplay style_disp(style_img, "style");
	style_img.normalize(0.0f, 1.0f);
	//preprocess(style_img);
	////////////////////////////////////////////////////////////////
#endif
	const int numLayers = 1;
	const char *targetLayerName[numLayers] = { end };
	const float layerStyleWeight[numLayers] = { 1.0f };
	LayerInfo_t layerInfos[numLayers];


	*//*
	// find destination layer and first hidden layer ///////
	HiddenLayer* dstLayer = dynamic_cast<HiddenLayer*>(network->findLayer(end));
	if(!dstLayer) {
		cout << "could not find layer of name " << end << " ... " << endl;
		exit(-1);
	}
	*//*

	HiddenLayer* firstHiddenLayer = dynamic_cast<HiddenLayer*>(network->getInputLayer()->getNextLayers()[0]);
	if(!firstHiddenLayer) {
		cout << "cout not find first hidden layer ... " << endl;
		exit(-1);
	}
	///////////////////////////////////////////////////////////////


	const int width = content_img.width();
	const int height = content_img.height();
	const int channel = content_img.spectrum();
	//Util::printData(content_img.data(), height, width, channel, 1, "content_img:");
	//Util::printData(style_img.data(), height, width, channel, 1, "style_img:");


	// prepare random input image /////////////////////////
	CImg<DATATYPE> input_img(width, height, 1, channel, 0.0f);
	CImgDisplay process_disp(input_img, "reconstruction");
	input_img.noise(10);
	input_img.normalize(0.0f, 1.0f);
	//preprocess(input_img);
	///////////////////////////////////////////////////////////////


	// prepare network for input image ///////////////////////////////////////////////////
	//network->reshape(io_dim(width, height, channel, 1));
	network->shape(io_dim(height, width, channel, 1));
	readLayerInfo(network, numLayers, targetLayerName, layerInfos);

	io_dim inputLayerOutDim = network->getInputLayer()->getInDimension();
	//io_dim dstLayerOutDim = dstLayer->getOutDimension();
	int inputLayerOutSize = network->getInputLayer()->getInDimension().unitsize();
	//int dstLayerOutSize = dstLayerOutDim.unitsize();
	//const int N = dstLayerOutDim.channels;
	//const int M = dstLayerOutDim.rows*dstLayerOutDim.cols;
	///////////////////////////////////////////////////////////////////////////////////////////////////


	// feed forward content image and get output ////////////////////////////////////////////////////////////////////////////////
	DATATYPE *d_content;
	checkCudaErrors(cudaMalloc(&d_content, sizeof(DATATYPE)*content_img.size()));
	checkCudaErrors(cudaMemcpyAsync(d_content, content_img.data(), sizeof(DATATYPE)*content_img.size(), cudaMemcpyHostToDevice));

	DATATYPE *content_out[numLayers];
	for(int i = 0; i < numLayers; i++) {
		network->feedforward(d_content, targetLayerName[i]);
		content_out[i] = new DATATYPE[layerInfos[i].outSize];

		//checkCudaErrors(cudaMalloc(&d_content_out, sizeof(DATATYPE)*dstLayerOutSize));
		checkCudaErrors(cudaMemcpyAsync(content_out[i], layerInfos[i].layer->getOutput(), sizeof(DATATYPE)*layerInfos[i].outSize, cudaMemcpyDeviceToHost));
		Util::printData(content_out[i], layerInfos[i].out_dim.rows, layerInfos[i].out_dim.cols, 3, 1, "content_out:");
	}
	checkCudaErrors(cudaFree(d_content));
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


#ifdef STYLE_
	// prepare Al
	DATATYPE *d_style;
	checkCudaErrors(cudaMalloc(&d_style, sizeof(DATATYPE)*style_img.size()));
	checkCudaErrors(cudaMemcpyAsync(d_style, style_img.data(), sizeof(DATATYPE)*style_img.size(), cudaMemcpyHostToDevice));

	DATATYPE *style_out[numLayers];
	for(int l = 0; l < numLayers; l++) {
		network->feedforward(d_style, targetLayerName[l]);

		style_out[l] = new DATATYPE[layerInfos[l].N*layerInfos[l].N];
		DATATYPE *style_temp_out = new DATATYPE[layerInfos[l].outSize];
		checkCudaErrors(cudaMemcpyAsync(style_temp_out, layerInfos[l].layer->getOutput(), sizeof(DATATYPE)*layerInfos[l].outSize, cudaMemcpyDeviceToHost));
		Util::printData(style_temp_out, layerInfos[l].out_dim.rows, layerInfos[l].out_dim.cols, layerInfos[l].out_dim.channels, 1, "style_temp_out:");
		gramMatrix(style_temp_out, layerInfos[l].N, layerInfos[l].M, style_out[l]);
		//Cube<DATATYPE> style_out_arma(style_out, 1, N, N);
		//style_out_arma.print("style_out_aram:");
		Util::printData(style_out[l], layerInfos[l].N, layerInfos[l].N, 1, 1, "style_out:");
		delete [] style_temp_out;
	}
	checkCudaErrors(cudaFree(d_style));
#endif

	DATATYPE *d_input;
	//DATATYPE *d_input_acc;
	checkCudaErrors(cudaMalloc(&d_input, sizeof(DATATYPE)*input_img.size()));
	//checkCudaErrors(cudaMalloc(&d_input_acc, sizeof(DATATYPE)*input_img.size()));
	DATATYPE *content_loss[numLayers];
	DATATYPE* content_loss_rmo[numLayers];
	DATATYPE* style_loss[numLayers];
	DATATYPE* style_delta_temp[numLayers];
	DATATYPE *d_content_loss[numLayers];

	for(int l = 0; l < numLayers; l++) {
		content_loss[l] = new DATATYPE[layerInfos[l].outSize];
		content_loss_rmo[l] = new DATATYPE[layerInfos[l].outSize];
		style_loss[l] = new DATATYPE[layerInfos[l].N*layerInfos[l].N];
		style_delta_temp[l] = new DATATYPE[layerInfos[l].N*layerInfos[l].M];
		checkCudaErrors(cudaMalloc(&d_content_loss[l], sizeof(DATATYPE)*layerInfos[l].outSize));
	}

	const float negative_one = -1.0f;
	const float learning_rate = -0.01f;
	const float alpha = 0.1f;
	const float beta = 100.0f;

	while(true) {
		checkCudaErrors(cudaMemcpyAsync(d_input, input_img.data(), sizeof(DATATYPE)*input_img.size(), cudaMemcpyHostToDevice));

		for(int l = numLayers-1; l >= 0; l--) {
			// for random image input,
			// compute output for specified layer
			network->feedforward(d_input, targetLayerName[l]);
			// compute content loss
			checkCudaErrors(cudaMemcpyAsync(content_loss[l], layerInfos[l].layer->getOutput(), sizeof(DATATYPE)*layerInfos[l].outSize, cudaMemcpyDeviceToHost));

#ifdef STYLE_
			// content loss를 변경하기전에 style loss부터 계산.
			//Util::setPrint(true);
			Util::printData(content_loss[l], layerInfos[l].out_dim.rows, layerInfos[l].out_dim.cols, layerInfos[l].out_dim.channels, 1, "style_out:");
			gramMatrix(content_loss[l], layerInfos[l].N, layerInfos[l].M, style_loss[l]);
			Util::printData(style_out[l], layerInfos[l].N, layerInfos[l].N, 1, 1, "style_out:");
			Util::printData(style_loss[l], layerInfos[l].N, layerInfos[l].N, 1, 1, "style_loss:");
			// G-A
			for(int i = 0; i < layerInfos[l].N*layerInfos[l].N; i++) {
				style_loss[l][i] -= style_out[l][i];
			}

			// transpose(F)*(G-A)
			//Util::setPrint(true);
			Util::printData(style_loss[l], layerInfos[l].N, layerInfos[l].N, 1, 1, "G-A:");
			Util::printData(content_loss[l], layerInfos[l].out_dim.rows, layerInfos[l].out_dim.cols, layerInfos[l].out_dim.channels, 1, "F:");
			//Util::setPrint(false);

			for(int channel = 0; channel < layerInfos[l].out_dim.channels; channel++) {
				for(int row = 0; row < layerInfos[l].out_dim.rows; row++) {
					for(int col = 0; col < layerInfos[l].out_dim.cols; col++) {
						content_loss_rmo[l][row+col*layerInfos[l].out_dim.rows+channel*layerInfos[l].M] = content_loss[l][col+row*layerInfos[l].out_dim.cols+channel*layerInfos[l].M];
					}
				}
			}
			//Util::printData(content_loss_rmo, dstLayerOutDim.rows, dstLayerOutDim.cols, dstLayerOutDim.channels, 1, "F_temp:");
			Cube<DATATYPE> F_temp(content_loss_rmo[l], 1, layerInfos[l].M, layerInfos[l].N);
			Mat<DATATYPE> G_A(style_loss[l], layerInfos[l].N, layerInfos[l].N);
			Mat<DATATYPE> F;
			for(int i = 0; i < F_temp.n_slices; i++) {
				F = join_cols(F, F_temp.slice(i));
			}
			//F.print("F:");
			//G_A.print("G-A:");

			Mat<DATATYPE> result = (F.t()*G_A).t();
			Util::printData(result.mem, layerInfos[l].N, layerInfos[l].M, 1, 1, "F_temp:");

			for(int i = 0; i < F.n_elem; i++) {
				if(F.mem[i] <= 0) result.memptr()[i] = 0.0f;
			}
			Util::printData(result.mem, layerInfos[l].N, layerInfos[l].M, 1, 1, "F_temp:");

			//DATATYPE* style_delta = new DATATYPE[N*M];
			const float coef = beta/(layerInfos[l].N*layerInfos[l].M*layerInfos[l].N*layerInfos[l].M);					// too small coef.
			//cout << "coef: " << coef << endl;
			result *= coef;

			for(int channel = 0; channel < layerInfos[l].out_dim.channels; channel++) {
				for(int m = 0; m < layerInfos[l].M; m++) {
					int row = m % layerInfos[l].out_dim.rows;
					int col = m / layerInfos[l].out_dim.rows;
					style_delta_temp[l][col+row*layerInfos[l].out_dim.cols+channel*layerInfos[l].M] =
							result.mem[m*layerInfos[l].N+channel];
				}
			}
			//Util::setPrint(true);
			Util::printData(style_delta_temp[l], layerInfos[l].out_dim.rows, layerInfos[l].out_dim.cols, layerInfos[l].out_dim.channels, 1, "style_delta_temp:");
			Util::setPrint(false);

#endif

			//DATATYPE* style_delta = new DATATYPE[N*M];
			//checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(dstLayerOutSize),
			//		&negative_one, d_content_out, 1, d_content_loss, 1));

			for(int i = 0; i < layerInfos[l].N*layerInfos[l].M; i++) {
				if(l == numLayers-1) {
					if(content_loss[l][i] > 0) content_loss[l][i] -= content_out[l][i];
					else content_loss[l][i] = 0;

					content_loss[l][i] = alpha*content_loss[l][i];// + style_delta_temp[i];
#ifdef STYLE_
					// style delta + content delta
					content_loss[l][i] += layerStyleWeight[l]*style_delta_temp[l][i];
#endif
				} else {
					content_loss[l][i] = layerStyleWeight[l]*style_delta_temp[l][i];
				}
			}

			Util::printData(content_loss[l], layerInfos[l].out_dim.rows, layerInfos[l].out_dim.cols, 3, 1, "content_loss:");
			checkCudaErrors(cudaMemcpyAsync(d_content_loss[l], content_loss[l], sizeof(DATATYPE)*layerInfos[l].outSize, cudaMemcpyHostToDevice));

			// back propagation 전에 error들을 합해준다.

			// back propagate content loss
			layerInfos[l].layer->backpropagation(0, d_content_loss[l]);

			//Util::setPrint(true);
			Util::printDeviceData(d_input, height, width, channel, 1, "input_img:");
			Util::printDeviceData(firstHiddenLayer->getDeltaInput(), 224, 224, 3, 1, "g:");
			// add g to input image
			checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(inputLayerOutSize),
					&learning_rate, firstHiddenLayer->getDeltaInput(), 1, d_input, 1));
			Util::printDeviceData(d_input, height, width, channel, 1, "d_input:");

#ifndef STYLE_
			break;
#endif

		}

		// visualize result
		checkCudaErrors(cudaMemcpyAsync(input_img.data(), d_input, sizeof(DATATYPE)*inputLayerOutSize, cudaMemcpyDeviceToHost));
		Util::printData(input_img.data(), height, width, channel, 1, "input_img:");
		//clipImage(input_img);

		CImg<DATATYPE> temp_src(input_img);
		//deprocess(temp_src);

		process_disp.resize(temp_src, true).display(temp_src.normalize(0, 255));
		cout << "reconstruction ... " << endl;
	}

	//DATATYPE *d_style;
	//checkCudaErrors(cudaMalloc(&d_style, sizeof(DATATYPE)*style_img.size()));
	//checkCudaErrors(cudaMemcpyAsync(d_style, style_img.data(), sizeof(DATATYPE)*style_img.size(), cudaMemcpyHostToDevice));

	while(!content_disp.is_closed()) {
		content_disp.wait();
	}

	while(!process_disp.is_closed()) {
		process_disp.wait();
	}



}


void ArtisticStyle::gramMatrix(DATATYPE* f, const int N, const int M, DATATYPE* g) {

	DATATYPE expectation[N];
	for(int i = 0; i < N; i++) {
		expectation[i] = 0.0f;
		for(int j = 0; j < M; j++) {
			expectation[i] += f[j+i*M];
		}
		expectation[i] /= M;
	}

	for(int i = 0; i < N; i++) {
		for(int j = 0; j < N; j++) {
			g[j*N+i] = expectation[i] * expectation[j];
		}
	}

	*//*
	// column major order !!!
	for(int i = 0; i < N; i++) {							// for row
		for(int j = 0; j < N; j++) {						// for column
			g[j*N+i] = 0.0;
			for(int k = 0; k < M; k++) {
				//g[j*N+i] += (f[k*N+i]*f[k*N+j]);		// row가 N개이므로 stride for column은 N
				g[j*N+i] += (f[k+i*M]*f[k+j*M]);		// row가 N개이므로 stride for column은 N
			}
		}
	}
	*//*
}







*/



template class ArtisticStyle<float>;

//#endif









