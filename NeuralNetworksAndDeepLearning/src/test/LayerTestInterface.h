/*
 * LayerTestInterface.h
 *
 *  Created on: Feb 13, 2017
 *      Author: jkim
 */

#ifndef LAYERTESTINTERFACE_H_
#define LAYERTESTINTERFACE_H_

template <typename Dtype>
class LayerTestInterface {
public:
	void setUp() = 0;
	void cleanUp() = 0;
	void forwardTest() = 0;
	void backwardTest() = 0;
};



#endif /* LAYERTESTINTERFACE_H_ */
