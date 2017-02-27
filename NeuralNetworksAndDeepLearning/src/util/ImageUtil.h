/**
 * @file ImageUtil.h
 * @date 2017-02-27
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef IMAGEUTIL_H
#define IMAGEUTIL_H 

#include "common.h"

template<typename Dtype>
class ImageUtil {
public: 
    ImageUtil() {}
    virtual ~ImageUtil() {}

    static void showImage(const Dtype* data, int nthImage, int channel, int row, int col);
};

#endif /* IMAGEUTIL_H */
