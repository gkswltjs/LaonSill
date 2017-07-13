/*
 * MultiLabelDataInputLayer.cpp
 *
 *  Created on: Jul 12, 2017
 *      Author: jkim
 */

#include "MultiLabelDataInputLayer.h"
#include "PropMgmt.h"
#include "SysLog.h"
#include "IO.h"

using namespace std;

template <typename Dtype>
MultiLabelDataInputLayer<Dtype>::MultiLabelDataInputLayer()
: InputLayer<Dtype>(),
  dataReader(SLPROP(Input, source)) {
	this->type = Layer<Dtype>::MultiLabelDataInput;
}

template <typename Dtype>
MultiLabelDataInputLayer<Dtype>::~MultiLabelDataInputLayer() {

}


template <typename Dtype>
void MultiLabelDataInputLayer<Dtype>::reshape() {
	if (this->_inputData.size() < 1) {
		for (uint32_t i = 0; i < SLPROP_BASE(output).size(); i++) {
			SLPROP_BASE(input).push_back(SLPROP_BASE(output)[i]);
			this->_inputData.push_back(this->_outputData[i]);
		}
	}
	Layer<Dtype>::_adjustInputShape();
	Datum* datum = this->dataReader.peekNextData();

	const uint32_t inputSize = this->_inputData.size();
	for (uint32_t i = 0; i < inputSize; i++) {
		if (!Layer<Dtype>::_isInputShapeChanged(i))
			continue;

		// data
		if (i == 0) {
			vector<uint32_t> dataShape =
			{SNPROP(batchSize), uint32_t(datum->channels), uint32_t(datum->height),
					uint32_t(datum->width)};
			this->_inputData[0]->reshape(dataShape);
			this->_inputShape[0] = dataShape;
		}
		// label
		else if (i == 1) {
			vector<uint32_t> labelShape = {SNPROP(batchSize),
					(uint32_t)SLPROP(MultiLabelDataInput, labelCount), 1, 1};
			this->_inputData[1]->reshape(labelShape);
			this->_inputShape[1] = labelShape;
		}
	}
}



template <typename Dtype>
void MultiLabelDataInputLayer<Dtype>::feedforward() {
	reshape();
	load_batch();
}

template <typename Dtype>
void MultiLabelDataInputLayer<Dtype>::feedforward(unsigned int baseIndex, const char* end) {
	reshape();
	load_batch();
}

template <typename Dtype>
void MultiLabelDataInputLayer<Dtype>::load_batch() {
	vector<float> mean = SLPROP(Input, mean);
	bool hasMean = false;
	if (mean.size() > 0) {
		hasMean = true;
	}

	for (int item_id = 0; item_id < SNPROP(batchSize); item_id++) {
		int offset = this->_inputData[0]->offset(item_id);
		Dtype* output_data = this->_inputData[0]->mutable_host_data();
		output_data += offset;

		Datum* datum = this->dataReader.getNextData();

		const string& data = datum->data;
		const int datum_channels = datum->channels;
		const int datum_height = datum->height;
		const int datum_width = datum->width;

		int height = datum_height;
		int width = datum_width;

		int h_off = 0;
		int w_off = 0;

		// DATA TRANSFORM ////////////////////////////////////////////////////////////////////
		// expand mean to follow real number of image channels
		if (hasMean) {
			SASSERT(mean.size() == 1 || mean.size() == datum_channels,
					"Specify either 1 mean value or as many as channels: %d", datum_channels);
			if (datum_channels > 1 && mean.size() == 1) {
				// Replicate the mean for simplicity
				for (int c = 1; c < datum_channels; c++) {
					mean.push_back(mean[0]);
				}
			}
		}

		// apply mean and scale
		const float scale = SLPROP(Input, scale);
		Dtype datum_element;
		int top_index, data_index;
		for (int c = 0; c < datum_channels; c++) {
			for (int h = 0; h < height; h++) {
				for (int w = 0; w < width; w++) {
					data_index = (c * datum_height + h_off + h) * datum_width + w_off + w;
					top_index = (c * height + h) * width + w;
					datum_element =
							static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));

					if (hasMean) {
						output_data[top_index] = (datum_element - mean[c]) * scale;
					} else {
						output_data[top_index] = datum_element * scale;
					}
				}
			}
		}
		//////////////////////////////////////////////////////////////////////////////////////

		Dtype* output_label = this->_inputData[1]->mutable_host_data();
		const int labelCount = SLPROP(MultiLabelDataInput, labelCount);
		output_label[item_id * labelCount + datum->label] = Dtype(1);
		for (int labelIdx = 0; labelIdx < datum->float_data.size(); labelIdx) {
			output_label[item_id * labelCount + (int)datum->float_data[labelIdx]] = Dtype(1);
		}
	}
}


template <typename Dtype>
int MultiLabelDataInputLayer<Dtype>::getNumTrainData() {
	return this->dataReader.getNumData();
}

template <typename Dtype>
int MultiLabelDataInputLayer<Dtype>::getNumTestData() {
	return 0;
}

template <typename Dtype>
void MultiLabelDataInputLayer<Dtype>::shuffleTrainDataSet() {

}








/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* MultiLabelDataInputLayer<Dtype>::initLayer() {
    MultiLabelDataInputLayer* layer = new MultiLabelDataInputLayer<Dtype>();
    return (void*)layer;
}

template<typename Dtype>
void MultiLabelDataInputLayer<Dtype>::destroyLayer(void* instancePtr) {
    MultiLabelDataInputLayer<Dtype>* layer = (MultiLabelDataInputLayer<Dtype>*)instancePtr;
    delete layer;
}

template<typename Dtype>
void MultiLabelDataInputLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
	// XXX
	if (isInput) {
		SASSERT0(false);
	} else {
		SASSERT0(index < 2);
	}

    MultiLabelDataInputLayer<Dtype>* layer = (MultiLabelDataInputLayer<Dtype>*)instancePtr;
    if (!isInput) {
        SASSERT0(layer->_outputData.size() == index);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool MultiLabelDataInputLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    MultiLabelDataInputLayer<Dtype>* layer = (MultiLabelDataInputLayer<Dtype>*)instancePtr;
    layer->reshape();

    if (SNPROP(miniBatch) == 0) {
		int trainDataNum = layer->getNumTrainData();
		if (trainDataNum % SNPROP(batchSize) == 0) {
			SNPROP(miniBatch) = trainDataNum / SNPROP(batchSize);
		} else {
			SNPROP(miniBatch) = trainDataNum / SNPROP(batchSize) + 1;
		}
		WorkContext::curPlanInfo->miniBatchCount = SNPROP(miniBatch);
	}

    return true;
}

template<typename Dtype>
void MultiLabelDataInputLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
	MultiLabelDataInputLayer<Dtype>* layer = (MultiLabelDataInputLayer<Dtype>*)instancePtr;
	layer->feedforward();
}

template<typename Dtype>
void MultiLabelDataInputLayer<Dtype>::backwardTensor(void* instancePtr) {
    // do nothing
}

template<typename Dtype>
void MultiLabelDataInputLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template class MultiLabelDataInputLayer<float>;
