/**
 * @file CelebAInputLayer.cpp
 * @date 2017-02-20
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <sys/types.h>
#include <dirent.h>
#include <fcntl.h>

#include <opencv2/opencv.hpp>

#include <boost/range/algorithm.hpp>

#include "common.h"
#include "CelebAInputLayer.h"
#include "InputLayer.h"
#include "NetworkConfig.h"
#include "ColdLog.h"
#include "SysLog.h"

using namespace std;

#define CELEBAINPUTLAYER_LOG        0

const int CELEBA_IMAGE_CHANNEL = 3;
const int CELEBA_IMAGE_ROW = 218;
const int CELEBA_IMAGE_COL = 178;

//const int CELEBA_CENTER_CROP_LEN = 108;

template<typename Dtype>
CelebAInputLayer<Dtype>::CelebAInputLayer() {
    initialize("", false, -1, false, -1, -1);
}

template<typename Dtype>
CelebAInputLayer<Dtype>::CelebAInputLayer(const string name, string imageDir, bool cropImage,
    int cropLen, bool resizeImage, int resizedImageRow, int resizedImageCol) :
    InputLayer<Dtype>(name) {
    initialize(imageDir, cropImage, cropLen, resizeImage, resizedImageRow, resizedImageCol);
}

template<typename Dtype>
CelebAInputLayer<Dtype>::CelebAInputLayer(Builder* builder) : InputLayer<Dtype>(builder) {
	initialize(builder->_imageDir, builder->_cropImage, builder->_cropLen,
        builder->_resizeImage, builder->_resizedImageRow, builder->_resizedImageCol);
}

template<typename Dtype>
CelebAInputLayer<Dtype>::~CelebAInputLayer() {
    if (this->images != NULL) {
        free(this->images); 
    }
}

template <typename Dtype>
void CelebAInputLayer<Dtype>::reshape() {
    int batchSize = this->networkConfig->_batchSize;

	if (this->images == NULL) {
        fillImagePaths();

        unsigned long allocSize = 
            (unsigned long)sizeof(Dtype) * 
            (unsigned long)this->imageRow * 
            (unsigned long)this->imageCol * 
            (unsigned long)this->imageChannel * 
            (unsigned long)batchSize;

        this->images = (Dtype*)malloc(allocSize);
        SASSERT0(this->images != NULL);
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

#if CELEBAINPUTLAYER_LOG
    printf("<%s> layer' output-0 has reshaped as: %dx%dx%dx%d\n",
        this->name.c_str(), batchSize, this->imageChannel, this->imageRow, this->imageCol);
#endif

    loadImages(this->currentBatchIndex);

    int inputImageSize = this->imageChannel * this->imageRow * this->imageCol * batchSize;

    this->_inputData[0]->set_device_with_host_data(this->images, 0, inputImageSize);
}

template<typename Dtype>
void CelebAInputLayer<Dtype>::loadPixels(cv::Mat image, int imageIndex) {
    unsigned long offset = (unsigned long)imageIndex * 
        (this->imageRow * this->imageCol * this->imageChannel);

    // XXX: find better method T_T
    // Red
    for (int row = 0; row < this->imageRow; row++) {
        for (int col = 0; col < this->imageCol; col++) {
            this->images[offset] = (Dtype)image.at<cv::Vec3b>(row, col)[2] / 127.5 - 1.0;
            offset++;
        }
    }

    // Green
    for (int row = 0; row < this->imageRow; row++) {
        for (int col = 0; col < this->imageCol; col++) {
            this->images[offset] = (Dtype)image.at<cv::Vec3b>(row, col)[1] / 127.5 - 1.0;
            offset++;
        }
    }

    // Blue
    for (int row = 0; row < this->imageRow; row++) {
        for (int col = 0; col < this->imageCol; col++) {
            this->images[offset] = (Dtype)image.at<cv::Vec3b>(row, col)[0] / 127.5 - 1.0;
            offset++;
        }
    }
}

template<typename Dtype>
void CelebAInputLayer<Dtype>::fillImagePaths() {
    vector<string> filePath;
    struct dirent *entry;
    DIR *dp;

    dp = opendir(this->imageDir.c_str());
    if (dp == NULL) {
        COLD_LOG(ColdLog::ERROR, true, "opendir() is failed. errno=%d", errno); 
        SASSERT0("opendir() is failed.");
    }

    while (entry = readdir(dp)) {
        string filename(entry->d_name);
        if (filename.find(".jpg") != string::npos) {
            this->imageIndexes.push_back(this->imagePaths.size());
            this->imagePaths.push_back(this->imageDir + "/" + filename);
        }
    }

    closedir(dp);
}

template<typename Dtype>
void CelebAInputLayer<Dtype>::loadImages(int baseIdx) {
    int batchSize = this->networkConfig->_batchSize;

    // (1) load jpeg
    for (int i = 0; i < batchSize; i++) {
        int index = i + baseIdx;
        if (index >= this->imagePaths.size())
            break;

        int shuffledIndex = this->imageIndexes[index];
        string imagePath = this->imagePaths[shuffledIndex];

        cv::Mat image;
        image = cv::imread(imagePath, CV_LOAD_IMAGE_COLOR);

        // XXX: 좀더 general 하게 만들자.
        //int imageCols = image.cols;
        //int imageRows = image.rows;
        //int imageChannels = image.channels();
        //SASSERT(imageCols == CELEBA_IMAGE_COL, "col : %d", imageCols);
        //SASSERT(imageRows == CELEBA_IMAGE_ROW, "row : %d", imageRows);
        //SASSERT(imageChannels == CELEBA_IMAGE_CHANNEL, "channel : %d", imageChannels);

        cv::Mat croppedImage;
        if (this->cropImage) {
            cv::Rect roi;
            roi.x = (CELEBA_IMAGE_COL - this->cropLen) / 2;
            roi.y = (CELEBA_IMAGE_ROW - this->cropLen) / 2;
            roi.width = this->cropLen;
            roi.height = this->cropLen;
            croppedImage = image(roi);

            if (!this->resizeImage) {
                loadPixels(croppedImage, i);
            }
        }

        if (this->resizeImage) {
            cv::Mat resizedImage;

            if (this->cropImage) {
                cv::resize(croppedImage, resizedImage, cv::Size(this->imageRow, this->imageCol));
            } else {
                cv::resize(image, resizedImage, cv::Size(this->imageRow, this->imageCol));
            }

            loadPixels(resizedImage, i);
        }
    }
}

template<typename Dtype>
void CelebAInputLayer<Dtype>::shuffleImages() {
    // FIXME: move it to other source.
#if 0
    srand(time(NULL)); 
    boost::range::random_shuffle(this->imageIndexes);
#endif
}

template<typename Dtype>
void CelebAInputLayer<Dtype>::feedforward() {
	//Layer<Dtype>::feedforward();
	cout << "unsupported ... " << endl;
	exit(1);
}

template<typename Dtype>
void CelebAInputLayer<Dtype>::feedforward(const uint32_t baseIndex, const char* end) {
    this->currentBatchIndex = baseIndex;    // FIXME: ...
    reshape();
}

template<typename Dtype>
void CelebAInputLayer<Dtype>::initialize(string imageDir, bool cropImage, int cropLen,
    bool resizeImage, int resizedImageRow, int resizedImageCol) {
    this->type = Layer<Dtype>::CelebAInput;
    this->imageDir = imageDir;
    this->cropImage = cropImage;
    this->resizeImage = resizeImage;

    this->imageRow = CELEBA_IMAGE_ROW;
    this->imageCol = CELEBA_IMAGE_COL;

    if (cropImage) {
        this->imageRow = cropLen;
        this->imageCol = cropLen;
        this->cropLen = cropLen;
    } 
    
    if (resizeImage) {
        this->imageRow = resizedImageRow;
        this->imageCol = resizedImageCol;
    }

    this->imageChannel = CELEBA_IMAGE_CHANNEL;

    this->images = NULL;
    this->currentBatchIndex = 0;
}

template<typename Dtype>
int CelebAInputLayer<Dtype>::getNumTrainData() {
    if (this->images == NULL) {
        reshape();
    }
    return this->imagePaths.size();
}

template<typename Dtype>
int CelebAInputLayer<Dtype>::getNumTestData() {
    return 0;
}

template<typename Dtype>
void CelebAInputLayer<Dtype>::shuffleTrainDataSet() {
    if (this->images == NULL) {
        reshape();
    }
    shuffleImages();
}

template class CelebAInputLayer<float>;
