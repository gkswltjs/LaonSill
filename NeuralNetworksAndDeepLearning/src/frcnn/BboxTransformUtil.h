/*
 * BboxTransformUtil.h
 *
 *  Created on: Nov 14, 2016
 *      Author: jkim
 */

#ifndef BBOXTRANSFORMUTIL_H_
#define BBOXTRANSFORMUTIL_H_


#include "frcnn_common.h"



struct BboxTransformUtil {
public:
	static void bboxTransofrm(const std::vector<std::vector<uint32_t>>& ex_rois,
			const std::vector<std::vector<uint32_t>>& gt_rois, std::vector<std::vector<float>>& result) {
		assert(ex_rois.size() == gt_rois.size());

		float ex_width, ex_height, ex_ctr_x, ex_ctr_y;
		float gt_width, gt_height, gt_ctr_x, gt_ctr_y;

		const uint32_t numRois = ex_rois.size();
		for (uint32_t i = 0; i < numRois; i++) {
			 ex_width = ex_rois[i][2] - ex_rois[i][0] + 1.0f;
			 ex_height = ex_rois[i][3] - ex_rois[i][1] + 1.0f;
			 ex_ctr_x = ex_rois[i][0] + 0.5f * ex_width;
			 ex_ctr_y = ex_rois[i][1] + 0.5f * ex_height;

			 gt_width = gt_rois[i][2] - gt_rois[i][0] + 1.0f;
			 gt_height = gt_rois[i][3] - gt_rois[i][1] + 1.0f;
			 gt_ctr_x = gt_rois[i][0] + 0.5f * gt_width;
			 gt_ctr_y = gt_rois[i][1] + 0.5f * gt_height;

			 // result[i][0] for label
			 result[i][1] = (gt_ctr_x - ex_ctr_x) / ex_width;
			 result[i][2] = (gt_ctr_y - ex_ctr_y) / ex_height;
			 result[i][3] = std::log(gt_width / ex_width);
			 result[i][4] = std::log(gt_height / ex_height);
		}
	}
};




#endif /* BBOXTRANSFORMUTIL_H_ */
