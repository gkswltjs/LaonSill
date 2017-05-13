/*
 * ssd_common.h
 *
 *  Created on: Apr 28, 2017
 *      Author: jkim
 */

#ifndef SSD_COMMON_H_
#define SSD_COMMON_H_

#include <iostream>

// The normalized bounding box [0, 1] w.r.t. the input image size
class NormalizedBBox {
public:
	float xmin;
	float ymin;
	float xmax;
	float ymax;
	int label;
	bool difficult;
	float score;
	float size;

	NormalizedBBox() {
		this->xmin = 0.f;
		this->ymin = 0.f;
		this->xmax = 0.f;
		this->ymax = 0.f;
		this->label = 0;
		this->difficult = false;
		this->score = 0.f;
		this->size = 0.f;
	}

	void print() {
		std::cout << "\txmin: " 		<< this->xmin		<< std::endl;
		std::cout << "\tymin: " 		<< this->ymin		<< std::endl;
		std::cout << "\txmax: " 		<< this->xmax		<< std::endl;
		std::cout << "\tymax: " 		<< this->ymax		<< std::endl;
		std::cout << "\tlabel: " 		<< this->label		<< std::endl;
		std::cout << "\tdifficult: "	<< this->difficult	<< std::endl;
		std::cout << "\tscore: " 		<< this->score		<< std::endl;
		std::cout << "\tsize: "			<< this->size		<< std::endl;
	}
};






#endif /* SSD_COMMON_H_ */
