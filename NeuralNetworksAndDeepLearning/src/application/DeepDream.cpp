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
}

DeepDream::~DeepDream() {}


void printImage(const char *head, CImg<DATATYPE>& img) {
	const DATATYPE *ptr = img.data(0);

	int width = std::min(5, img.width());
	int height = std::min(5, img.height());

	cout << head << "-" << img.width() << "x" << img.height() << "x" << img.spectrum() << endl;
	for(int i = 0; i < height; i++) {
		for(int j = 0; j < width; j++) {
			cout << ptr[i*img.width()+j] << ", ";
		}
		cout << endl;
	}
}


void DeepDream::deepdream() {

	CImg<DATATYPE> image(base_img);
	image.normalize(0.0, 1.0);
	printImage("image", image);
	//CImgDisplay main_disp(image, "input image");

	preprocess(image);

	vector<CImg<DATATYPE>> octaves(octave_n);
	//vector<DATATYPE *> d_octaves(octave_n);
	for(int i = 0; i < octave_n; i++) {
		octaves[i] = image;
		//checkCudaErrors(cudaMalloc(&d_octaves[i], sizeof(DATATYPE)*image.size()));
		//checkCudaErrors(cudaMemcpyAsync(d_octaves[i], octaves[i].data(0), sizeof(DATATYPE)*image.size(), cudaMemcpyHostToDevice));
		cout << "image size for octave " << i << "-width: " << image.width() << ", height: " << image.height() << ", spectrum: " << image.spectrum() << endl;

		image.resize(image.width()/octave_scale, image.height()/octave_scale, -100, -100, 5);
	}

	CImg<DATATYPE> src;
	printImage("src", src);
	//CImgDisplay process_disp(src, "proccess");

	CImg<DATATYPE> detail(octaves[octave_n-1], "xyzc", 0.0);
	printImage("detail", detail);

	for(int octave_index = octave_n-1; octave_index >= 0; octave_index--) {
		CImg<DATATYPE>& octave_base = octaves[octave_index];
		printImage("octave_base", octave_base);
		int w = octave_base.width();
		int h = octave_base.height();
		if(octave_index < octave_n-1) {
			int w1 = detail.width();
			int h1 = detail.height();
			detail.resize(w/(double)w1, h/(double)h1, -100, -100, 5);
			printImage("detail", detail);
		}

		network->reshape(io_dim(h, w, octave_base.spectrum(), 1));
		src = octave_base + detail;
		printImage("src", src);


		DATATYPE *d_src;
		checkCudaErrors(cudaMalloc(&d_src, sizeof(DATATYPE)*src.size()));
		checkCudaErrors(cudaMemcpyAsync(d_src, src.data(0), sizeof(DATATYPE)*src.size(), cudaMemcpyHostToDevice));

		for(int i = 0; i < iter_n; i++) {
			make_step(d_src);

			//visualization
			deprocess(src);

			// adjust image contrast if clipping is disabled
			if(!clip) {
				//vis = vis*(255.0/
			}
			CImgDisplay process_disp(src, "proccess");
			cout << "octave: " << octave_index << ", iter: " << i << ", end: " << end << ", dim: " << endl;

		}

		checkCudaErrors(cudaMemcpyAsync(src.data(), d_src, sizeof(DATATYPE)*src.size(), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaFree(d_src));

		detail = src - octave_base;
		printImage("detail", detail);

		break;
	}

	deprocess(src);

	//for(int i = 0; i < octave_n; i++) {
	//	checkCudaErrors(cudaFree(d_octaves[i]));
	//}

	//while(!main_disp.is_closed()) {
	//	main_disp.wait();
	//}
}


void DeepDream::make_step(DATATYPE *d_src, const char *end, float step_size, float jitter) {

	//jitter

	network->feedforward(d_src, end);

	// objective



	//network->backprop()


}

void DeepDream::objective_L2() {

}

void DeepDream::preprocess(CImg<DATATYPE>& img) {

}

void DeepDream::deprocess(CImg<DATATYPE>& img) {

}


