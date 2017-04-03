/**
 * @file EtriInputLayer.cpp
 * @date 2017-03-28
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <fcntl.h>

#include <iostream>

#include <opencv2/opencv.hpp>

#include "common.h"
#include "EtriInputLayer.h"
#include "InputLayer.h"
#include "NetworkConfig.h"
#include "ColdLog.h"
#include "SysLog.h"
#include "ImageUtil.h"

using namespace std;

#define ETRIINPUTLAYER_LOG        0
// FIXME: 다른 방식으로 file path를 얻자. 
#define ETRI_TOP1000_KEYWORD_FILENAME       "top1000keywords.txt"
#define ETRI_KEYWORD_FILENAME               "keywords.txt"

const int ETRIDATA_IMAGE_CHANNEL = 3;
const int ETRIDATA_LABEL_COUNT = 1000;

template<typename Dtype>
EtriInputLayer<Dtype>::EtriInputLayer() {
    initialize("", -1, -1, true);
}

template<typename Dtype>
EtriInputLayer<Dtype>::EtriInputLayer(const string name, string imageDir,
    int resizedImageRow, int resizedImageCol, bool train) :
    InputLayer<Dtype>(name) {
    initialize(imageDir, resizedImageRow, resizedImageCol, train);
}

template<typename Dtype>
EtriInputLayer<Dtype>::EtriInputLayer(Builder* builder) : InputLayer<Dtype>(builder) {
	initialize(builder->_imageDir, builder->_resizedImageRow, builder->_resizedImageCol,
        builder->_train);
}

template<typename Dtype>
EtriInputLayer<Dtype>::~EtriInputLayer() {
    if (this->images != NULL) {
        free(this->images); 
    }

    if (this->labels != NULL) {
        free(this->labels);
    }
}

template<typename Dtype>
void EtriInputLayer<Dtype>::prepareKeywordMap() {
    int index = 0;
    string top1000KeywordFilePath = this->imageDir + "/" + ETRI_TOP1000_KEYWORD_FILENAME;

    ifstream input(top1000KeywordFilePath.c_str());
    string line;
    
    SASSERT0(input.is_open());

    while (input.good()) {
        getline(input, line);
        if (line == "")
            break;

        this->keywordMap[line] = index;
        index++;
    }

    input.close();


#if 1
    map <string, int>::iterator iter;
    cout << "Keyword Map" << endl;
    cout << "=================================================================" << endl;
    for (iter = this->keywordMap.begin(); iter != this->keywordMap.end(); iter++) {
        cout << iter->second << " : " << iter->first << endl;
    }
    cout << "=================================================================" << endl;

#endif


    SASSERT((this->keywordMap.size() <= ETRIDATA_LABEL_COUNT),
        "keyword count of etri data should be less than %d but %d.",
        ETRIDATA_LABEL_COUNT, (int)this->keywordMap.size());
}

template<typename Dtype>
void EtriInputLayer<Dtype>::registerData(string filePath) {
    // (1) read keywords
    string keywordFilePath = filePath + "/" + ETRI_KEYWORD_FILENAME;

    ifstream input(keywordFilePath.c_str());
    string line;
    
    SASSERT0(input.is_open());

    vector<int> labels;

    while (input.good()) {
        getline(input, line);
        if (this->keywordMap.find(line) != this->keywordMap.end()) {
            labels.push_back(this->keywordMap[line]);
        } 
    }

    input.close();

    if (labels.size() == 0)
        return;

    // (2) walk directory
    struct dirent *entry;
    DIR *dp;

    dp = opendir(filePath.c_str());
    SASSERT0(dp != NULL);

    vector<string> imageFileList;

    while ((entry = readdir(dp))) {
        string imageFileName(entry->d_name);
        string imageFilePath = filePath + "/" + imageFileName;

        if (imageFilePath.find(".jpg") != string::npos) {
            imageFileList.push_back(imageFilePath); 
        }
    }

    closedir(dp);

    // (3) register data
    //   1st data => test data
    //   others   => training data
    //   FIXME: inefficient..
   
    if (imageFileList.size() < 4)
        return;

    for (int i = 0; i < imageFileList.size(); i++) {
        EtriData newData;
        newData.filePath = imageFileList[i];

        for (int j = 0; j < labels.size(); j++) {
            newData.labels.push_back(labels[j]);
        }

        if (i % 4 == 0)
            this->testData.push_back(newData);        
        else
            this->trainData.push_back(newData);
    }
}


template<typename Dtype>
void EtriInputLayer<Dtype>::prepareData() {
    struct dirent *entry;
    DIR *dp;

    dp = opendir(this->imageDir.c_str());
    SASSERT0(dp != NULL);

    int step = 0;

    struct stat s;
    while ((entry = readdir(dp))) {
        string fileName(entry->d_name);
        if (fileName == "." || fileName == "..")
            continue;

        string filePath = this->imageDir + "/" + fileName;

        if (stat (filePath.c_str(), &s) == 0) {
            if (s.st_mode & S_IFDIR) {
                registerData(filePath);
            }
        }

        step++;

#if 0
        if (step > 2000)
            break;
#endif
    }

    closedir(dp);
}

template<typename Dtype>
void EtriInputLayer<Dtype>::loadPixels(cv::Mat image, int imageIndex) {
    unsigned long offset = (unsigned long)imageIndex * 
        (this->imageRow * this->imageCol * this->imageChannel);

    // XXX: find better method T_T
    // Red
    for (int row = 0; row < this->imageRow; row++) {
        for (int col = 0; col < this->imageCol; col++) {
            this->images[offset] = (Dtype)image.at<cv::Vec3b>(row, col)[2] / 127.5 - 1.0;
            //this->images[offset] = (Dtype)image.at<cv::Vec3b>(row, col)[2];
            offset++;
        }
    }

    // Green
    for (int row = 0; row < this->imageRow; row++) {
        for (int col = 0; col < this->imageCol; col++) {
            this->images[offset] = (Dtype)image.at<cv::Vec3b>(row, col)[1] / 127.5 - 1.0;
            //this->images[offset] = (Dtype)image.at<cv::Vec3b>(row, col)[1];
            offset++;
        }
    }

    // Blue
    for (int row = 0; row < this->imageRow; row++) {
        for (int col = 0; col < this->imageCol; col++) {
            this->images[offset] = (Dtype)image.at<cv::Vec3b>(row, col)[0] / 127.5 - 1.0;
            //this->images[offset] = (Dtype)image.at<cv::Vec3b>(row, col)[0];
            offset++;
        }
    }
}

template<typename Dtype>
void EtriInputLayer<Dtype>::loadImages(int batchIndex) {
    int batchSize = this->networkConfig->_batchSize;
    int baseIndex = batchIndex;

    for (int i = 0; i < batchSize; i++) {
        int index = baseIndex + i;

        if (this->train) {
            SASSERT(index < this->trainData.size(),
                "index sholud be less than data count. index=%d, data count=%d",
                index, (int)this->trainData.size());
        } else {
            SASSERT(index < this->testData.size(),
                "index sholud be less than data count. index=%d, data count=%d",
                index, (int)this->testData.size());
        }

        cv::Mat image;
        string imagePath;

        // XXX: 
        if (this->train)
            imagePath = this->trainData[index].filePath;
        else
            imagePath = this->testData[index].filePath;

        image = cv::imread(imagePath, CV_LOAD_IMAGE_COLOR);

        int imageChannels = image.channels();
        SASSERT(imageChannels == ETRIDATA_IMAGE_CHANNEL, "channel : %d", imageChannels);

        cv::Mat resizedImage;
        cv::resize(image, resizedImage, cv::Size(this->imageRow, this->imageCol));
        loadPixels(resizedImage, i);

        
    }
}

template<typename Dtype>
void EtriInputLayer<Dtype>::loadLabels(int batchIndex) {
    int batchSize = this->networkConfig->_batchSize;
    int baseIndex = batchIndex;

    int totalSize = sizeof(Dtype) * ETRIDATA_LABEL_COUNT * batchSize;
    memset(this->labels, 0x00, totalSize);

    for (int i = 0; i < batchSize; i++) {
        int index = baseIndex + i;

        if (this->train) {
            SASSERT(index < this->trainData.size(),
                "index sholud be less than data count. index=%d, data count=%d",
                index, (int)this->trainData.size());
        } else {
            SASSERT(index < this->testData.size(),
                "index sholud be less than data count. index=%d, data count=%d",
                index, (int)this->testData.size());
        }

        vector<int> curLabels;

        // XXX: 
        if (this->train)
            curLabels = this->trainData[index].labels;
        else
            curLabels = this->testData[index].labels;

        for (int j = 0; j < curLabels.size(); j++) {
            int pos = curLabels[j];
            SASSERT0(pos < ETRIDATA_LABEL_COUNT);
            this->labels[i * ETRIDATA_LABEL_COUNT + pos] = 1.0;
        }
    }
}

template <typename Dtype>
void EtriInputLayer<Dtype>::reshape() {
    int batchSize = this->networkConfig->_batchSize;

    if (this->images == NULL) {
        unsigned long allocSize = 
            (unsigned long)sizeof(Dtype) * 
            (unsigned long)this->imageRow * 
            (unsigned long)this->imageCol * 
            (unsigned long)this->imageChannel * 
            (unsigned long)batchSize;

        this->images = (Dtype*)malloc(allocSize);

        SASSERT0(this->images != NULL);
        // prepare keyword map

        prepareKeywordMap();

        // prepare training & test data
        prepareData();

        SASSERT0(this->labels == NULL);
        unsigned long labelAllocSize = 
            (unsigned long)sizeof(Dtype) * 
            (unsigned long)ETRIDATA_LABEL_COUNT *
            (unsigned long)batchSize;

        this->labels = (Dtype*)malloc(labelAllocSize);
        SASSERT0(this->labels != NULL);
	}

	if (this->_inputData.size() < 1) {
		for (uint32_t i = 0; i < this->_outputs.size(); i++) {
			this->_inputs.push_back(this->_outputs[i]);
			this->_inputData.push_back(this->_outputData[i]);
		}
    }

	Layer<Dtype>::_adjustInputShape();

    this->_inputShape[0][0] = batchSize;
    this->_inputShape[0][1] = this->imageChannel;
    this->_inputShape[0][2] = this->imageRow;
    this->_inputShape[0][3] = this->imageCol;

    this->_inputData[0]->reshape(this->_inputShape[0]);

#if ETRIINPUTLAYER_LOG
    printf("<%s> layer' output-0 has reshaped as: %dx%dx%dx%d\n",
        this->name.c_str(), batchSize, this->imageChannel, this->imageRow, this->imageCol);
#endif

    this->_inputShape[1][0] = batchSize;
    this->_inputShape[1][1] = 1;
    this->_inputShape[1][2] = ETRIDATA_LABEL_COUNT;
    this->_inputShape[1][3] = 1;

    this->_inputData[1]->reshape(this->_inputShape[1]);

#if ETRIINPUTLAYER_LOG
    printf("<%s> layer' output-1 has reshaped as: %dx%dx%dx%d\n",
        this->name.c_str(), batchSize, 1, ETRIDATA_LABEL_COUNT, 1);
#endif

    loadImages(this->currentBatchIndex);
    loadLabels(this->currentBatchIndex);

    int inputImageSize = this->imageChannel * this->imageRow * this->imageCol * batchSize;
    int inputLabelSize = ETRIDATA_LABEL_COUNT * batchSize;

    this->_inputData[0]->set_device_with_host_data(this->images, 0, inputImageSize);
    this->_inputData[1]->set_device_with_host_data(this->labels, 0, inputLabelSize);
}

template<typename Dtype>
void EtriInputLayer<Dtype>::shuffleImages() {
    // FIXME: move it to other source.
#if 0
    srand(time(NULL)); 
    boost::range::random_shuffle(this->imageIndexes);
#endif
}

template<typename Dtype>
void EtriInputLayer<Dtype>::feedforward() {
	//Layer<Dtype>::feedforward();
	cout << "unsupported ... " << endl;
	exit(1);
}

template<typename Dtype>
void EtriInputLayer<Dtype>::feedforward(const uint32_t baseIndex, const char* end) {
    this->currentBatchIndex = baseIndex;    // FIXME: ...
    reshape();
}

template<typename Dtype>
void EtriInputLayer<Dtype>::initialize(string imageDir, int resizedImageRow,
    int resizedImageCol, bool train) {
    this->type = Layer<Dtype>::EtriInput;
    this->imageDir = imageDir;
    this->imageRow = resizedImageRow;
    this->imageCol = resizedImageCol;
    this->imageChannel = ETRIDATA_IMAGE_CHANNEL;
    this->train = train;

    this->images = NULL;
    this->labels = NULL;
    this->currentBatchIndex = 0;
}

template<typename Dtype>
int EtriInputLayer<Dtype>::getNumTrainData() {
    if (this->images == NULL) {
        reshape();
    }
    return this->trainData.size();
}

template<typename Dtype>
int EtriInputLayer<Dtype>::getNumTestData() {
    if (this->images == NULL) {
        reshape();
    }

    return this->testData.size();
}

template<typename Dtype>
void EtriInputLayer<Dtype>::shuffleTrainDataSet() {
    if (this->images == NULL) {
        reshape();
    }
    shuffleImages();
}

template class EtriInputLayer<float>;
