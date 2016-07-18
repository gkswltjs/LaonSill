/*
 * DeepDream.h
 *
 *  Created on: 2016. 7. 16.
 *      Author: jhkim
 */

#ifndef DEEPDREAM_H_
#define DEEPDREAM_H_


#include "../network/Network.h"
#include <CImg.h>

using namespace cimg_library;


class DeepDream {
public:
	DeepDream(Network *network, const char *base_img, UINT iter_n=10, UINT octave_n=4,
			double octave_scale=1.4, const char *end="inception_4c/output", bool clip=true);
	virtual ~DeepDream();

	void deepdream();

private:
	void make_step(DATATYPE *src, const char *end, float step_size=1.5, float jitter=32);
	void objective_L2();
	void preprocess(CImg<DATATYPE>& img);
	void deprocess(CImg<DATATYPE>& img);

	Network *network;
	char base_img[256];
	UINT iter_n;
	UINT octave_n;
	double octave_scale;
	char end[256];
	bool clip;
};


#endif /* DEEPDREAM_H_ */
