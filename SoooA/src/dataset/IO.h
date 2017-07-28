/*
 * IO.h
 *
 *  Created on: Jun 29, 2017
 *      Author: jkim
 */

#ifndef IO_H_
#define IO_H_

#include <string>
#include <opencv2/core/core.hpp>
#include <map>

#include "Datum.h"

bool ReadImageToDatum(const std::string& filename, const std::vector<int>& label,
		const int height, const int width, const int min_dim, const int max_dim,
		const bool channel_separated, const bool is_color, const std::string& encoding,
		class Datum* datum);

bool ReadRichImageToAnnotatedDatum(const std::string& filename, const std::string& labelname,
		const int height, const int width, const int min_dim, const int max_dim,
		const bool is_color, const std::string& encoding, const std::string& labeltype,
		const std::map<std::string, int>& name_to_label, AnnotatedDatum* anno_datum);

/*
bool ReadImageToDatum(const std::string& filename, const int label, const int height,
		const int width, const bool is_color, const std::string& encoding, Datum* datum) {
	//return ReadImageToDatum(filename, label, height, width, 0, 0, is_color, encoding, datum);
	return false;
}
*/

cv::Mat ReadImageToCVMat(const std::string& filename, const int height, const int width,
		const int min_dim, const int max_dim, const bool is_color);


void CVMatToDatum(const cv::Mat& cv_img, const bool channel_separated, Datum* datum);

bool ReadFileToDatum(const std::string& filename, const int label, Datum* datum);

void EncodeCVMatToDatum(const cv::Mat& cv_img, const std::string& encoding, Datum* datum);




cv::Mat DecodeDatumToCVMat(const Datum* datum, bool is_color, bool channel_separated);


bool ReadXMLToAnnotatedDatum(const std::string& labelname, const int img_height,
		const int img_width, const std::map<std::string, int>& name_to_label,
		AnnotatedDatum* anno_datum);



void GetImageSize(const std::string& filename, int* height, int* width);




template <typename Dtype>
void CheckCVMatDepthWithDtype(const cv::Mat& im);

template <typename Dtype>
void ConvertHWCToCHW(const int channels, const int height, const int width, const Dtype* src,
		Dtype* dst);
template <typename Dtype>
void ConvertHWCCVToCHW(const cv::Mat& im, Dtype* dst);

template <typename Dtype>
void ConvertHWCToHWC(const int channels, const int height, const int width, const Dtype* src,
		Dtype* dst);
template <typename Dtype>
void ConvertHWCCVToHWC(const cv::Mat& im, Dtype* dst);
template <typename Dtype>
void ConvertCHWToHWC(const int channels, const int height, const int width, const Dtype* src,
		Dtype* dst);
void ConvertCHWDatumToHWC(const Datum* datum, uchar* dst);

template <typename Dtype>
void PrintImageData(const int channels, const int height, const int width, const Dtype* ptr,
		bool hwc);
void PrintCVMatData(const cv::Mat& mat);
void PrintDatumData(const Datum* datum, bool hwc);


#endif /* IO_H_ */
