/*
 * ArtisticStyle.h
 *
 *  Created on: 2016. 7. 22.
 *      Author: jhkim
 */

#ifndef ARTISTICSTYLE_H_
#define ARTISTICSTYLE_H_

#include "../network/Network.h"
#include <CImg.h>

using namespace cimg_library;

#ifndef GPU_MODE

class ArtisticStyle {
public:
	ArtisticStyle(Network *network);
	virtual ~ArtisticStyle();

	void style(const char* content_img_path, const char* style_img_path,
			const char* end);

private:
	void gramMatrix(DATATYPE* f, const int N, const int M, DATATYPE* g);
	void preprocess(CImg<DATATYPE>& img);
	void deprocess(CImg<DATATYPE>& img);
	void clipImage(CImg<DATATYPE>& img);

	Network *network;
	DATATYPE mean[3];
};

#endif

#endif /* ARTISTICSTYLE_H_ */
