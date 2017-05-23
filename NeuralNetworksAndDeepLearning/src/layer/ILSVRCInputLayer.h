/**
 * @file ILSVRCInputLayer.h
 * @date 2017-04-19
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef ILSVRCINPUTLAYER_H
#define ILSVRCINPUTLAYER_H 

#include <vector>
#include <string>

#include <opencv2/opencv.hpp>

#include "common.h"
#include "InputLayer.h"
#include "Layer.h"

typedef struct ILSVRCMeta_s {
    std::string filePath;
    int         classID;
} ILSVRCMeta;

template<typename Dtype>
class ILSVRCInputLayer : public InputLayer<Dtype> {
public: 
	class Builder : public InputLayer<Dtype>::Builder {
	public:
        std::string _imageDir;
        int         _resizeImage;
        int         _resizedImageRow;
        int         _resizedImageCol;

		Builder() {
			this->type = Layer<Dtype>::ILSVRCInput;
            this->_imageDir = "";
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
		virtual Builder* propDown(const std::vector<bool>& propDown) {
			InputLayer<Dtype>::Builder::propDown(propDown);
			return this;
		}
        Builder* imageDir(std::string imageDir) {
            this->_imageDir = imageDir;
            return this;
        }
        Builder* resizeImage(int row, int col) {
            this->_resizeImage = true;
            this->_resizedImageRow = row;
            this->_resizedImageCol = col;
            return this;
        }
		Layer<Dtype>* build() {
			return new ILSVRCInputLayer(this);
		}
	};

    ILSVRCInputLayer(const std::string& name);

	ILSVRCInputLayer(const std::string name, const std::string imageDir,
        bool resizeImage, int resizedImageRow, int resizedImageCol);
	ILSVRCInputLayer(Builder* builder);

    virtual ~ILSVRCInputLayer();

	void feedforward();
	using Layer<Dtype>::feedforward;
	void feedforward(const uint32_t baseIndex, const char* end=0);

    int getNumTrainData();
    int getNumTestData();
    void shuffleTrainDataSet();

	void reshape();

protected:
	void initialize(const std::string imageDir, bool resizeImage, int resizedImageRow,
        int resizedImageCol);

    std::string imageDir;
    bool        resizeImage;
    int         imageCount;

    int         imageRow;
    int         imageCol;
    int         imageChannel;

    void        fillMetas();
    void        loadImages(int baseIdx);
    void        loadLabels(int baseIdx);
    void        loadPixels(cv::Mat image, int imageIndex);
    void        shuffleImages();

    Dtype*      images;
    Dtype*      labels;
    std::vector<ILSVRCMeta>     metas;

    std::vector<int>            metaIndexes;
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
#endif /* ILSVRCINPUTLAYER_H */
