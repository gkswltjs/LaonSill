/**
 * @file VOCPascalInputLayer.cpp
 * @date 2017-04-18
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
#include "VOCPascalInputLayer.h"
#include "InputLayer.h"
#include "NetworkConfig.h"
#include "ColdLog.h"
#include "SysLog.h"

using namespace std;

#define VOCPASCALINPUTLAYER_LOG        0

const int VOCPASCAL_IMAGE_CHANNEL   = 3;
const int VOCPASCAL_IMAGE_ROW       = 448;
const int VOCPASCAL_IMAGE_COL       = 448;

const int VOCPASCAL_BOX_COUNT       = 1;
const int VOCPASCAL_CLASS_COUNT     = 20;
const int VOCPASCAL_BOX_ELEM_COUNT  = (6 + VOCPASCAL_CLASS_COUNT);
/*********************************************************************************
 * Label
 * +-------+-------+---+---+-------+--------+-----------+
 * | gridX | gridY | x | y | width | height | class(20) |
 * +-------+-------+---+---+-------+--------+-----------+
 */

const int VOCPASCAL_GRID_COUNT      = 7;

template<typename Dtype>
VOCPascalInputLayer<Dtype>::VOCPascalInputLayer() {
    initialize("", false, -1, -1);
}

template<typename Dtype>
VOCPascalInputLayer<Dtype>::VOCPascalInputLayer(const string name, string imageDir,
    bool resizeImage, int resizedImageRow, int resizedImageCol) :
    InputLayer<Dtype>(name) {
    initialize(imageDir, resizeImage, resizedImageRow, resizedImageCol);
}

template<typename Dtype>
VOCPascalInputLayer<Dtype>::VOCPascalInputLayer(Builder* builder) : 
    InputLayer<Dtype>(builder) {
	initialize(builder->_imageDir, builder->_resizeImage, builder->_resizedImageRow,
        builder->_resizedImageCol);
}

template<typename Dtype>
VOCPascalInputLayer<Dtype>::~VOCPascalInputLayer() {
    if (this->images != NULL) {
        free(this->images); 
    }
}

template <typename Dtype>
void VOCPascalInputLayer<Dtype>::reshape() {
    int batchSize = this->networkConfig->_batchSize;

	if (this->images == NULL) {
        SASSERT0(this->labels == NULL);
        fillMetas();

        unsigned long allocImageSize = 
            (unsigned long)sizeof(Dtype) * 
            (unsigned long)this->imageRow * 
            (unsigned long)this->imageCol * 
            (unsigned long)this->imageChannel * 
            (unsigned long)batchSize;

        this->images = (Dtype*)malloc(allocImageSize);
        SASSERT0(this->images != NULL);

        unsigned long allocLabelSize = 
            (unsigned long)sizeof(Dtype) * 
            (unsigned long)VOCPASCAL_BOX_COUNT *
            (unsigned long)VOCPASCAL_BOX_ELEM_COUNT *
            (unsigned long)batchSize;

        this->labels = (Dtype*)malloc(allocLabelSize);
        SASSERT0(this->labels != NULL);
	} else {
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

    this->_inputShape[1][0] = batchSize;
    this->_inputShape[1][1] = 1;
    this->_inputShape[1][2] = VOCPASCAL_BOX_COUNT * VOCPASCAL_BOX_ELEM_COUNT;
    this->_inputShape[1][3] = 1;
    this->_inputData[1]->reshape(this->_inputShape[1]);

#if VOCPASCALINPUTLAYER_LOG
    printf("<%s> layer' output-0 has reshaped as: %dx%dx%dx%d\n",
        this->name.c_str(), batchSize, this->imageChannel, this->imageRow, this->imageCol);
    printf("<%s> layer' output-1 has reshaped as: %dx%dx%dx%d\n",
        this->name.c_str(), batchSize, 1, VOCPASCAL_BOX_COUNT * VOCPASCAL_BOX_ELEM_COUNT, 1);
#endif

    loadImages(this->currentBatchIndex);
    loadLabels(this->currentBatchIndex);

    int inputImageSize = this->imageChannel * this->imageRow * this->imageCol * batchSize;
    this->_inputData[0]->set_device_with_host_data(this->images, 0, inputImageSize);

    int inputLabelSize = VOCPASCAL_BOX_COUNT * VOCPASCAL_BOX_ELEM_COUNT * batchSize;
    this->_inputData[1]->set_device_with_host_data(this->labels, 0, inputLabelSize);
}

template<typename Dtype>
void VOCPascalInputLayer<Dtype>::loadPixels(cv::Mat image, int imageIndex) {
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

#define VOCPASCAL_METAFILE_NAME        "pascal_voc.txt"

template<typename Dtype>
void VOCPascalInputLayer<Dtype>::fillMetas() {
    string filePath = this->imageDir + "/" + VOCPASCAL_METAFILE_NAME;
    FILE *fp = fopen(filePath.c_str(), "r");
    SASSERT0(fp != NULL);

    char* line = NULL;
    size_t len = 0;
    ssize_t read;

    char imagePath[1024];
    int boxCount = 0;
    int imageWidth;
    int imageHeight;
    VOCPascalMeta meta;

    int metaIndex = 0;

    while ((read = getline(&line, &len, fp)) != -1) {
        if (boxCount == 0) {
            int ret = sscanf(line, "%s %d", imagePath, &boxCount);
            SASSERT0(ret == 2);
            meta.imagePath = imagePath;

            cv::Mat image = cv::imread(meta.imagePath, CV_LOAD_IMAGE_COLOR);
            imageWidth = image.cols;
            imageHeight = image.rows;
        } else {
            int xmin, ymin, xmax, ymax, classID;
            int ret = sscanf(line, "%d %d %d %d %d", &xmin, &ymin, &xmax, &ymax, &classID);
            SASSERT0(ret == 5);

            float centerX = ((float)xmin + (float)xmax) / 2.0 / (float)imageWidth;
            float centerY = ((float)ymin + (float)ymax) / 2.0 / (float)imageHeight;

            meta.gridX = (int)(centerX * (float)VOCPASCAL_GRID_COUNT);
            SASSERT0((meta.gridX >= 0) && (meta.gridX < VOCPASCAL_GRID_COUNT));

            meta.gridY = (int)(centerY * (float)VOCPASCAL_GRID_COUNT);
            SASSERT0((meta.gridY >= 0) && (meta.gridY < VOCPASCAL_GRID_COUNT));

            meta.x = (centerX - (float)meta.gridX / (float)VOCPASCAL_GRID_COUNT) * 
                (float)VOCPASCAL_GRID_COUNT;
            meta.y = (centerY - (float)meta.gridY / (float)VOCPASCAL_GRID_COUNT) * 
                (float)VOCPASCAL_GRID_COUNT;

            SASSERT0(xmax > xmin);
            SASSERT0(ymax > ymin);

            meta.width = ((float)xmax - (float)xmin) / (float)imageWidth;
            meta.height = ((float)ymax - (float)ymin) / (float)imageHeight;

            meta.classID = classID;

            this->metas.push_back(meta);
            this->metaIndexes.push_back(metaIndex);
            metaIndex++;
            boxCount--;
        }
    }
   
    if (line)
        free(line);

    fclose(fp);
}

template<typename Dtype>
void VOCPascalInputLayer<Dtype>::loadImages(int baseIdx) {
    int batchSize = this->networkConfig->_batchSize;

    for (int i = 0; i < batchSize; i++) {
        int index = i + baseIdx;
        if (index >= this->metas.size())
            break;

        int shuffledIndex = this->metaIndexes[index];
        string imagePath = this->metas[shuffledIndex].imagePath;

        cv::Mat image;
        image = cv::imread(imagePath, CV_LOAD_IMAGE_COLOR);

        // XXX: 좀더 general 하게 만들자.
        cv::Mat resizedImage;
        cv::resize(image, resizedImage, cv::Size(this->imageRow, this->imageCol));

        loadPixels(resizedImage, i);
    }
}

#define EPSILON     0.001       // to solve converting issue int to float(double)

template<typename Dtype>
void VOCPascalInputLayer<Dtype>::loadLabels(int baseIdx) {
    int batchSize = this->networkConfig->_batchSize;

    for (int i = 0; i < batchSize; i++) {
        int index = i + baseIdx;
        if (index >= this->metas.size())
            break;

        int shuffledIndex = this->metaIndexes[index];
       
        VOCPascalMeta *meta = &this->metas[shuffledIndex];
        this->labels[i * VOCPASCAL_BOX_ELEM_COUNT + 0] = (Dtype)meta->gridX + EPSILON;
        this->labels[i * VOCPASCAL_BOX_ELEM_COUNT + 1] = (Dtype)meta->gridY + EPSILON;
        this->labels[i * VOCPASCAL_BOX_ELEM_COUNT + 2] = (Dtype)meta->x;
        this->labels[i * VOCPASCAL_BOX_ELEM_COUNT + 3] = (Dtype)meta->y;
        this->labels[i * VOCPASCAL_BOX_ELEM_COUNT + 4] = (Dtype)meta->width;
        this->labels[i * VOCPASCAL_BOX_ELEM_COUNT + 5] = (Dtype)meta->height;
        
        for (int j = 0; j < VOCPASCAL_CLASS_COUNT; j++) {
            this->labels[i * VOCPASCAL_BOX_ELEM_COUNT + 6 + j] = 0.0;
        }

        SASSERT0(meta->classID < VOCPASCAL_CLASS_COUNT);
        SASSERT0(meta->classID >= 0);
        this->labels[i * VOCPASCAL_BOX_ELEM_COUNT + 6 + meta->classID] = 1.0;
    }
}

template<typename Dtype>
void VOCPascalInputLayer<Dtype>::shuffleImages() {
    // FIXME: move it to other source.
#if 0
    srand(time(NULL)); 
    boost::range::random_shuffle(this->metaIndexes);
#endif
}

template<typename Dtype>
void VOCPascalInputLayer<Dtype>::feedforward() {
	//Layer<Dtype>::feedforward();
	cout << "unsupported ... " << endl;
	exit(1);
}

template<typename Dtype>
void VOCPascalInputLayer<Dtype>::feedforward(const uint32_t baseIndex, const char* end) {
    this->currentBatchIndex = baseIndex;    // FIXME: ...
    reshape();
}

template<typename Dtype>
void VOCPascalInputLayer<Dtype>::initialize(string imageDir, bool resizeImage,
    int resizedImageRow, int resizedImageCol) {
    this->type = Layer<Dtype>::VOCPascalInput;
    this->imageDir = imageDir;
    this->resizeImage = resizeImage;

    this->imageRow = VOCPASCAL_IMAGE_ROW;
    this->imageCol = VOCPASCAL_IMAGE_COL;

    if (resizeImage) {
        this->imageRow = resizedImageRow;
        this->imageCol = resizedImageCol;
    }

    this->imageChannel = VOCPASCAL_IMAGE_CHANNEL;

    this->images = NULL;
    this->labels = NULL;

    this->currentBatchIndex = 0;
}

template<typename Dtype>
int VOCPascalInputLayer<Dtype>::getNumTrainData() {
    if (this->images == NULL) {
        reshape();
    }
    return this->metas.size();
}

template<typename Dtype>
int VOCPascalInputLayer<Dtype>::getNumTestData() {
    return 0;
}

template<typename Dtype>
void VOCPascalInputLayer<Dtype>::shuffleTrainDataSet() {
    if (this->images == NULL) {
        reshape();
    }
    shuffleImages();
}

template class VOCPascalInputLayer<float>;
