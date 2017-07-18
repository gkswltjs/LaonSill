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

#include "Datum.h"

bool ReadImageToDatum(const std::string& filename, const std::vector<int>& label,
		const int height, const int width, const int min_dim, const int max_dim,
		const bool channel_separated, const bool is_color, const std::string& encoding,
		class Datum* datum);

/*
bool ReadImageToDatum(const std::string& filename, const int label, const int height,
		const int width, const bool is_color, const std::string& encoding, Datum* datum) {
	//return ReadImageToDatum(filename, label, height, width, 0, 0, is_color, encoding, datum);
	return false;
}
*/

cv::Mat ReadImageToCVMat(const std::string& filename, const int height, const int width,
		const int min_dim, const int max_dim, const bool is_color);

void CVMatToDatum(const cv::Mat& cv_img, const bool channel_separated, class Datum* datum);

cv::Mat DecodeDatumToCVMat(const class Datum* datum, bool is_color);



#endif /* IO_H_ */
