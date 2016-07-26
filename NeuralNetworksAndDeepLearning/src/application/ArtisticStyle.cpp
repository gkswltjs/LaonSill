/*
 * ArtisticStyle.cpp
 *
 *  Created on: 2016. 7. 22.
 *      Author: jhkim
 */

#include "ArtisticStyle.h"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <driver_types.h>
#include <cstdlib>
#include <iostream>
#include <armadillo>

#include "../cuda/Cuda.h"
#include "../layer/HiddenLayer.h"
#include "../layer/InputLayer.h"
#include "../layer/Layer.h"
#include "../layer/LayerConfig.h"
#include "../network/Network.h"
#include "../Util.h"

using namespace std;
using namespace cimg_library;
using namespace arma;


struct LayerInfo_t {
	HiddenLayer* layer;
	io_dim out_dim;
	int outSize;
	int N;
	int M;
};




#define STYLE_

ArtisticStyle::ArtisticStyle(Network *network) {
	this->network = network;

	for(int i = 0; i < 3; i++) {
		mean[i] = network->getDataSetMean(i);
	}
}

ArtisticStyle::~ArtisticStyle() {
	// TODO Auto-generated destructor stub
}


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


	/*
	// find destination layer and first hidden layer ///////
	HiddenLayer* dstLayer = dynamic_cast<HiddenLayer*>(network->findLayer(end));
	if(!dstLayer) {
		cout << "could not find layer of name " << end << " ... " << endl;
		exit(-1);
	}
	*/

	HiddenLayer* firstHiddenLayer = dynamic_cast<HiddenLayer*>(network->getInputLayer()->getNextLayers()[0].next_layer);
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

	/*
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
	*/
}




void ArtisticStyle::preprocess(CImg<DATATYPE>& img) {
	DATATYPE *data_ptr = img.data();
	const int height = img.height();
	const int width = img.width();
	const int channel = img.spectrum();

	for(int c = 0; c < channel; c++) {
		for(int h = 0; h < height; h++) {
			for(int w = 0; w < width; w++) {
				data_ptr[w+h*width+c*height*width] -= mean[c];
			}
		}
	}
}

void ArtisticStyle::deprocess(CImg<DATATYPE>& img) {

	DATATYPE *data_ptr = img.data();
	const int height = img.height();
	const int width = img.width();
	const int channel = img.spectrum();

	for(int c = 0; c < channel; c++) {
		for(int h = 0; h < height; h++) {
			for(int w = 0; w < width; w++) {
				data_ptr[w+h*width+c*height*width] += mean[c];
			}
		}
	}
}

void ArtisticStyle::clipImage(CImg<DATATYPE>& img) {
	DATATYPE *src_ptr = img.data();
	const int width = img.width();
	const int height = img.height();
	const int channel = img.spectrum();
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
}














