/*
 * InputLayer.cpp
 *
 *  Created on: 2016. 9. 12.
 *      Author: jhkim
 */


#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "InputLayer.h"
#include "NetworkConfig.h"
#include "ImagePackDataSet.h"
#include "LMDBDataSet.h"
#include "MockDataSet.h"
#include "Util.h"
#include "CudaUtils.h"

#define INPUTLAYER_LOG 1

using namespace std;

template <typename Dtype>
InputLayer<Dtype>::InputLayer()
: Layer<Dtype>() {
    _dataSet = NULL;
	initialize();
}

template <typename Dtype>
InputLayer<Dtype>::InputLayer(const string name)
: Layer<Dtype>(name) {
    _dataSet = NULL;
	initialize();
}

template <typename Dtype>
InputLayer<Dtype>::InputLayer(Builder* builder)
: Layer<Dtype>(builder) {

	this->_scale = builder->_scale;
	this->_dataMean = new Data<Dtype>("dataMean");

	if (builder->_sourceType == "ImagePack") {
		this->_dataSet = new ImagePackDataSet<Dtype>(
				builder->_source+"/train_data",
				builder->_source+"/train_label",
				builder->_numTrainPack,
				builder->_source+"/test_data",
				builder->_source+"/test_label",
				builder->_numTestPack);
		//this->_dataSet->setMean({0.13066047740});
		//this->_dataSet->setMean(builder->_mean);

		int numChannels = builder->_mean.size();
		this->_dataMean->reshape({1, 1, 1, numChannels});
		for (int i = 0; i < numChannels; i++) {
			this->_dataMean->mutable_host_data()[i] = builder->_mean[i];
		}

		this->_dataSet->load();

	} else if (builder->_sourceType == "LMDB") {
		this->_dataSet = new LMDBDataSet<Dtype>(
				builder->_source);
		int numChannels = builder->_mean.size();
		this->_dataMean->reshape({1, 1, 1, numChannels});
		for (int i = 0; i < numChannels; i++) {
			this->_dataMean->mutable_host_data()[i] = builder->_mean[i];
		}
		this->_dataSet->load();

	} else if (builder->_sourceType == "Mock") {
		this->_dataSet = new MockDataSet<Dtype>(
				4, 4, 3, 10, 10, 10
				);
		this->_dataSet->load();
	} else {
        _dataSet = NULL;
		//cout << "Unsuppored Input Source Type: " << builder->_sourceType;
		//exit(1);
	}

	initialize();
}

template <typename Dtype>
InputLayer<Dtype>::~InputLayer() {
	if (this->_dataMean) {
		delete this->_dataMean;
	}
}


template <typename Dtype>
void InputLayer<Dtype>::reshape() {
	// 입력 레이어는 출력 데이터를 입력 데이터와 공유
	// xxx: 레이어 명시적으로 초기화할 수 있는 위치를 만들어 옮길 것.
	if (this->_inputData.size() < 1) {
		for (uint32_t i = 0; i < this->_outputs.size(); i++) {
			this->_inputs.push_back(this->_outputs[i]);
			this->_inputData.push_back(this->_outputData[i]);
		}
	}

	Layer<Dtype>::_adjustInputShape();

	const uint32_t inputSize = this->_inputData.size();
	for (uint32_t i = 0; i < inputSize; i++) {
		if (!Layer<Dtype>::_isInputShapeChanged(i))
			continue;

		// 데이터
		if (i == 0) {
			uint32_t batches = this->networkConfig->_batchSize;
			uint32_t channels = this->_dataSet->getChannels();
			uint32_t rows = this->_dataSet->getRows();
			uint32_t cols = this->_dataSet->getCols();

			this->_inputShape[0][0] = batches;
			this->_inputShape[0][1] = channels;
			this->_inputShape[0][2] = rows;
			this->_inputShape[0][3] = cols;

			this->_inputData[0]->reshape(this->_inputShape[0]);

#if INPUTLAYER_LOG
			printf("<%s> layer' output-0 has reshaped as: %dx%dx%dx%d\n",
					this->name.c_str(), batches, channels, rows, cols);
#endif
		}
		// 레이블
		else if (i == 1) {
			uint32_t batches = this->networkConfig->_batchSize;
			uint32_t channels = 1;
			uint32_t rows = 1;
			uint32_t cols = 1;

			this->_inputShape[1][0] = batches;
			this->_inputShape[1][1] = channels;
			this->_inputShape[1][2] = rows;
			this->_inputShape[1][3] = cols;

			this->_inputData[1]->reshape({batches, channels, rows, cols});

#if INPUTLAYER_LOG
			printf("<%s> layer' output-1 has reshaped as: %dx%dx%dx%d\n",
					this->name.c_str(), batches, channels, rows, cols);
#endif
		}
	}

	/*
	this->in_dim.batches = this->networkConfig->_batchSize;
	this->in_dim.channels = this->_dataSet->getChannels();
	this->in_dim.rows = this->_dataSet->getRows();
	this->in_dim.cols = this->_dataSet->getCols();
	this->out_dim = this->in_dim;

	// 레이블 데이터가 있는 경우
	if (this->_outputs.size() > 1) {
		this->_inputData[1]->shape({this->in_dim.batches, 1, 1, 1});
	}

	if (recursive) {
		Layer<Dtype>::_shape();
	}
	//_shape();
	 */
}

template <typename Dtype>
void InputLayer<Dtype>::feedforward() {
	reshape();
	//cout << "unsupported ... " << endl;
	//exit(1);
	// do nothing
}


template <typename Dtype>
void InputLayer<Dtype>::feedforward(const uint32_t baseIndex, const char* end) {
	reshape();

	const vector<uint32_t>& inputShape = this->_inputShape[0];
	const uint32_t batches = inputShape[0];
	const uint32_t unitSize = Util::vecCountByAxis(inputShape, 1);

	//Data<Dtype>::printConfig = true;
	//SyncMem<Dtype>::printConfig = true;

	//cout.precision(0);
	if (this->networkConfig->_status == NetworkStatus::Train) {
		// data
		for (uint32_t i = 0; i < batches; i++) {
			const Dtype* ptr = _dataSet->getTrainDataAt(baseIndex+i);
			this->_inputData[0]->set_device_with_host_data(ptr, i*unitSize, unitSize);
			//this->_inputData[0]->print_data({}, false);

			if (this->_inputs.size() > 1) {
				const Dtype* ptr = _dataSet->getTrainLabelAt(baseIndex+i);
				this->_inputData[1]->set_device_with_host_data(ptr, i, 1);
				//cout << "label: " << ptr[0] << endl;
			}
			/*
			Data<float>* data = new Data<float>("");
			data->reshape({1, 3, 224, 224});
			float* data_ptr = data->mutable_host_data();
			for (int i = 0; i < 224*224*3; i++) {
				data_ptr[i] = float(ptr[i]) / 255;
			}
			data->transpose({0, 2, 3, 1});
			cv::Mat result(224, 224, CV_32FC3, data_ptr);
			cv::namedWindow("result");
			cv::imshow("result", result);
			cv::waitKey();
			delete data;
			*/
		}
		this->_inputData[0]->scale_device_data(this->_scale);
		//this->_inputData[0]->print_data({}, false);

		const uint32_t singleChannelSize = this->_inputData[0]->getCountByAxis(2);
		const Dtype* mean = this->_dataMean->device_data();

		for (uint32_t i = 0; i < batches; i++) {
			Dtype* data = this->_inputData[0]->mutable_device_data() + i*unitSize;
			soooa_sub_channel_mean(unitSize, singleChannelSize, mean, data);
		}

		//this->_inputData[0]->print_data({}, false);
		//Data<Dtype>::printConfig = true;
		//this->_inputData[0]->print_data("data");

		/*
		// label
		if (this->_inputs.size() > 1) {
			for (uint32_t i = 0; i < batches; i++) {
				const Dtype* ptr = _dataSet->getTrainLabelAt(baseIndex+i);
				this->_inputData[1]->set_device_with_host_data(ptr, i, 1);
			}
			//this->_inputData[1]->print_data("label");
		}
		*/

	} else if (this->networkConfig->_status == NetworkStatus::Test) {
		for(uint32_t i = 0; i < batches; i++) {
			const Dtype* ptr = _dataSet->getTestDataAt(baseIndex+i);
			this->_inputData[0]->set_device_with_host_data(ptr, i*unitSize, unitSize);

			if (this->_inputs.size() > 1) {
				const Dtype* ptr = _dataSet->getTestLabelAt(baseIndex+i);
				this->_inputData[1]->set_device_with_host_data(ptr, i, 1);
			}
		}
	} else {
        SASSERT(false, "Invalid network status =%d", this->networkConfig->_status);
    }
	//Layer<Dtype>::feedforward();


	Data<Dtype>::printConfig = false;
	SyncMem<Dtype>::printConfig = false;
}

template <typename Dtype>
void InputLayer<Dtype>::initialize() {
	this->type = Layer<Dtype>::Input;
}

template<typename Dtype>
int InputLayer<Dtype>::getNumTrainData() {
    return this->_dataSet->getNumTrainData();
}

template<typename Dtype>
int InputLayer<Dtype>::getNumTestData() {
    return this->_dataSet->getNumTestData();
}

template<typename Dtype>
void InputLayer<Dtype>::shuffleTrainDataSet() {
    return this->_dataSet->shuffleTrainDataSet();
}

template class InputLayer<float>;
