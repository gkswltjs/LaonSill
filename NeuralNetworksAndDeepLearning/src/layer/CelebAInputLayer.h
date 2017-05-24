/**
 * @file CelebAInputLayer.h
 * @date 2017-02-20
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef CELEBAINPUTLAYER_H
#define CELEBAINPUTLAYER_H 

#include <string>

#include <opencv2/opencv.hpp>

#include "common.h"
#include "InputLayer.h"
#include "Layer.h"

template <typename Dtype>
class CelebAInputLayer : public InputLayer<Dtype> {
public: 
	class Builder : public InputLayer<Dtype>::Builder {
	public:
        std::string _imageDir;
        bool        _cropImage;
        int         _cropLen;
        int         _resizeImage;
        int         _resizedImageRow;
        int         _resizedImageCol;

		Builder() {
			this->type = Layer<Dtype>::CelebAInput;
            this->_imageDir = "";
            this->_cropImage = false;
            this->_cropLen = 0;
            this->_resizeImage = false;
            this->_resizedImageRow = 0;
            this->_resizedImageCol = 0;
		}
		virtual Builder* name(const std::string name) {
			InputLayer<Dtype>::Builder::name(name);
			return this;
		}
		virtual Builder* id(uint32_t id) {
			InputLayer<Dtype>::Builder::id(id);
			return this;
		}
		virtual Builder* inputs(const std::vector<std::string>& inputs) {
			InputLayer<Dtype>::Builder::inputs(inputs);
			return this;
		}
		virtual Builder* outputs(const std::vector<std::string>& outputs) {
			InputLayer<Dtype>::Builder::outputs(outputs);
			return this;
		}
        Builder* imageDir(std::string imageDir) {
            this->_imageDir = imageDir;
            return this;
        }
        Builder* cropImage(int cropLen) {
            this->_cropImage = true;
            this->_cropLen = cropLen;
            return this;
        }
        Builder* resizeImage(int row, int col) {
            this->_resizeImage = true;
            this->_resizedImageRow = row;
            this->_resizedImageCol = col;
            return this;
        }
		Layer<Dtype>* build() {
			return new CelebAInputLayer(this);
		}
	};

    CelebAInputLayer();
    
	CelebAInputLayer(const std::string name, const std::string imageDir, bool cropImage,
        int cropLen, bool resizeImage, int resizedImageRow, int resizedImageCol);
    CelebAInputLayer(const std::string& name);
	CelebAInputLayer(Builder* builder);

    virtual ~CelebAInputLayer();

	void feedforward();
	using Layer<Dtype>::feedforward;
	void feedforward(const uint32_t baseIndex, const char* end=0);

    int getNumTrainData();
    int getNumTestData();
    void shuffleTrainDataSet();

	void reshape();

protected:
	void initialize(const std::string imageDir, bool cropImage, int cropLen,
        bool resizeImage, int resizedImageRow, int resizedImageCol);

    std::string imageDir;
    bool        cropImage;
    bool        resizeImage;
    int         imageCount;

    int         cropLen;
    int         imageRow;
    int         imageCol;
    int         imageChannel;

    void        fillImagePaths();
    void        loadImages(int baseIdx);
    void        loadPixels(cv::Mat image, int imageIndex);
    void        shuffleImages();

    std::vector<int> imageIndexes;
    std::vector<std::string> imagePaths;
    Dtype*      images; 
    int         currentBatchIndex;

public:
    /****************************************************************************
     * layer callback functions 
     ****************************************************************************/
    static void* initLayer();
    static void destroyLayer(void* instancePtr);
    static void setInOutTensor(void* instancePtr, void* tensorPtr, bool isInput, int index);
    static bool allocLayerTensors(void* instancePtr);
    static void forwardTensor(void* instancePtr, int miniBatchIndex);
    static void backwardTensor(void* instancePtr);
    static void learnTensor(void* instancePtr);
};
#endif /* CELEBAINPUTLAYER_H */
