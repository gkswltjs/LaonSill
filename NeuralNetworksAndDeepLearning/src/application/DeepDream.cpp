/*
 * DeepDream.cpp
 *
 *  Created on: 2016. 7. 16.
 *      Author: jhkim
 */

#include "DeepDream.h"



DeepDream::DeepDream(Network *network, const char *base_img, UINT iter_n, UINT octave_n,
		double octave_scale, const char *end, bool clip) {

	this->network = network;
	strcpy(this->base_img, base_img);
	this->iter_n = iter_n;
	this->octave_n = octave_n;
	this->octave_scale = octave_scale;
	strcpy(this->end, end);
	this->clip = clip;

	for(int i = 0; i < 3; i++) {
		mean[i] = network->getDataSetMean(i);
	}
}

DeepDream::~DeepDream() {}


void printImage(const char *head, DATATYPE *data, int w, int h, int c, bool force=false) {
	if(force || true) {
		int width = std::min(5, w);
		int height = std::min(5, h);

		cout << head << "-" << w << "x" << h << "x" << c << endl;
		for(int i = 0; i < height; i++) {
			for(int j = 0; j < width; j++) {
				cout << data[i*w+j] << ", ";
			}
			cout << endl;
		}
	}
}


void printImage(const char *head, CImg<DATATYPE>& img, bool force=false) {
	if(force || true) {
		printImage(head, img.data(), img.width(), img.height(), img.spectrum(), force);
	}
}





void DeepDream::deepdream() {
	CImg<DATATYPE> image(base_img);
	image.normalize(0.0, 1.0);
	preprocess(image);
	//printImage("preprocess", image);
	Util::printData(image.data(), image.height(), image.width(), image.spectrum(), 1, "image:");

	CImgDisplay main_disp(image, "input image");

	vector<CImg<DATATYPE>> octaves(octave_n);
	//vector<DATATYPE *> d_octaves(octave_n);
	for(int i = 0; i < octave_n; i++) {
		octaves[i] = image;
		cout << "image size for octave " << i << "-width: " << image.width() << ", height: " << image.height() << ", spectrum: " << image.spectrum() << endl;

		image.resize(image.width()/octave_scale, image.height()/octave_scale, -100, -100, 5);
	}

	CImg<DATATYPE> src;
	//printImage("src", src);
	CImgDisplay process_disp(src, "proccess");

	CImg<DATATYPE> detail(octaves[octave_n-1], "xyzc", 0.0);
	//printImage("detail", detail);

	for(int octave_index = octave_n-1; octave_index >= 0; octave_index--) {
		CImg<DATATYPE>& octave_base = octaves[octave_index];
		Util::printData(octave_base.data(), octave_base.height(), octave_base.width(), octave_base.spectrum(), 1, "octave_base:");

		int w = octave_base.width();
		int h = octave_base.height();
		if(octave_index < octave_n-1) {
			detail.resize(w, h, -100, -100, 5);
			//printImage("detail", detail);
			Util::printData(detail.data(), detail.height(), detail.width(), detail.spectrum(), 1, "detail:");
		}


		network->reshape(io_dim(h, w, octave_base.spectrum(), 1));

		src = octave_base + detail;
		//printImage("src", src, true);


		DATATYPE *d_src;
		checkCudaErrors(cudaMalloc(&d_src, sizeof(DATATYPE)*src.size()));
		for(int i = 0; i < iter_n; i++) {
			checkCudaErrors(cudaMemcpyAsync(d_src, src.data(), sizeof(DATATYPE)*src.size(), cudaMemcpyHostToDevice));


			Util::printData(src.data(), src.height(), src.width(), src.spectrum(), 1, "src:");
			Util::printDeviceData(d_src, src.height(), src.width(), src.spectrum(), 1, "d_src:");

			make_step(src.data(), d_src, end, 1.5);

			//unshift image
			if(clip) { clipImage(src); }

			Util::printData(src.data(), src.height(), src.width(), src.spectrum(), 1, "src_after_make_step:");
			// reconstruction된 이미지를 다시 normalize ...
			//src.normalize(0.0, 1.0);
			//Util::printData(src.data(), src.height(), src.width(), src.spectrum(), 1, "src_after_make_step_normalize:");

			// display용 cimg //////////////////////////////////////////////////////////////////////////////
			CImg<DATATYPE> temp_src(src);
			//visualization
			deprocess(temp_src);
			//temp_src.normalize(0.0, 1.0);
			//printImage("temp_src", temp_src);

			// adjust image contrast if clipping is disabled
			if(!clip) {
				//vis = vis*(255.0/
			}
			process_disp.resize(temp_src, true).display(temp_src).wait(20);
			cout << "octave: " << octave_index << ", iter: " << i << ", end: " << end << ", dim: " << endl;
			/////////////////////////////////////////////////////////////////////////////////////////////////////////////
		}
		//checkCudaErrors(cudaMemcpyAsync(src.data(), d_src, sizeof(DATATYPE)*src.size(), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaFree(d_src));

		detail = src - octave_base;
		Util::printData(detail.data(), detail.height(), detail.width(), detail.spectrum(), 1, "detail:");
	}

	deprocess(src);

	//for(int i = 0; i < octave_n; i++) {
	//	checkCudaErrors(cudaFree(d_octaves[i]));
	//}

	while(!process_disp.is_closed()) {
		process_disp.wait();
	}

	while(!main_disp.is_closed()) {
		main_disp.wait();
	}

}


void DeepDream::make_step(DATATYPE *src, DATATYPE *d_src, const char *end, float step_size, float jitter) {
	//jitter


	network->feedforward(d_src, end);
	HiddenLayer* dst = dynamic_cast<HiddenLayer*>(network->findLayer(end));
	if(!dst) {
		cout << "could not find layer of name " << end << " ... " << endl;
		exit(-1);
	}
	HiddenLayer* dst_nextLayer = dynamic_cast<HiddenLayer*>(dst->getNextLayers()[0].next_layer);
	if(!dst_nextLayer) {
		cout << "could not find next layer ... " << endl;
		exit(-1);
	}
	dst_nextLayer->setDeltaInput(dst->getOutput());
	dst->backpropagation(0, dst_nextLayer);
	HiddenLayer* firstHiddenLayer = dynamic_cast<HiddenLayer*>(network->getInputLayer()->getNextLayers()[0].next_layer);
	if(!firstHiddenLayer) {
		cout << "cout not find first hidden layer ... " << endl;
		exit(-1);
	}

	io_dim in_dim = network->getInputLayer()->getInDimension();
	int input_b_outsize = in_dim.batchsize();
	DATATYPE *g = new DATATYPE[input_b_outsize];
	checkCudaErrors(cudaMemcpyAsync(g, firstHiddenLayer->getDeltaInput(), sizeof(DATATYPE)*input_b_outsize, cudaMemcpyDeviceToHost));
	//firstHiddenLayer->getDeltaInput()

	Util::printData(g, in_dim.rows, in_dim.cols, in_dim.channels, 1, "g:");
	Util::printData(src, in_dim.rows, in_dim.cols, in_dim.channels, 1, "src:");

	float g_mean = 0.0f;
	for(int i = 0; i < input_b_outsize; i++) {
		g_mean += std::abs(g[i]);
	}
	g_mean /= input_b_outsize;

	for(int i = 0; i < input_b_outsize; i++) {
		src[i] += step_size/g_mean*g[i];
	}

	//printImage("src", src, in_dim.cols, in_dim.rows, in_dim.channels, true);
	Util::printData(src, in_dim.rows, in_dim.cols, in_dim.channels, 1, "src:");

	delete [] g;
}

void DeepDream::objective_L2() {

}

void DeepDream::preprocess(CImg<DATATYPE>& img) {
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

void DeepDream::deprocess(CImg<DATATYPE>& img) {

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

void DeepDream::clipImage(CImg<DATATYPE>& img) {
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

