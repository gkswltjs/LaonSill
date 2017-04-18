/**
 * @file VOCPascalInputLayer.h
 * @date 2017-04-18
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef VOCPASCALINPUTLAYER_H
#define VOCPASCALINPUTLAYER_H 

#include <vector>
#include <string>

#include <opencv2/opencv.hpp>

#include "common.h"
#include "InputLayer.h"
#include "Layer.h"

typedef struct VOCPascalMeta_s {
    std::string     imagePath;
    float           x;
    float           y;
    float           width;
    float           height;
    int             gridX;
    int             gridY;
    int             classID;
} VOCPascalMeta;

template<typename Dtype>
class VOCPascalInputLayer : public InputLayer<Dtype> {
public: 
	class Builder : public InputLayer<Dtype>::Builder {
	public:
		std::vector<uint32_t> _shape;
		std::string _source;
		std::string _sourceType;

        std::string _imageDir;
        int         _resizeImage;
        int         _resizedImageRow;
        int         _resizedImageCol;

		Builder() {
			this->type = Layer<Dtype>::CelebAInput;
            this->_imageDir = "";
            this->_resizeImage = false;
            this->_resizedImageRow = 0;
            this->_resizedImageCol = 0;
		}
		virtual Builder* shape(const std::vector<uint32_t>& shape) {
			this->_shape = shape;
			return this;
		}
		virtual Builder* source(const std::string& source) {
			this->_source = source;
			return this;
		}
		virtual Builder* sourceType(const std::string& sourceType) {
			this->_sourceType = sourceType;
			return this;
		}
		virtual Builder* name(const std::string name) {
			Layer<Dtype>::Builder::name(name);
			return this;
		}
		virtual Builder* id(uint32_t id) {
			Layer<Dtype>::Builder::id(id);
			return this;
		}
		virtual Builder* inputs(const std::vector<std::string>& inputs) {
			Layer<Dtype>::Builder::inputs(inputs);
			return this;
		}
		virtual Builder* outputs(const std::vector<std::string>& outputs) {
			Layer<Dtype>::Builder::outputs(outputs);
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
        }
		Layer<Dtype>* build() {
			return new VOCPascalInputLayer(this);
		}
	};

    VOCPascalInputLayer();

	VOCPascalInputLayer(const std::string name, const std::string imageDir,
        bool resizeImage, int resizedImageRow, int resizedImageCol);
	VOCPascalInputLayer(Builder* builder);

    virtual ~VOCPascalInputLayer();

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

    Dtype*                      images;
    Dtype*                      labels;

    std::vector<VOCPascalMeta>  metas;
    std::vector<int>            metaIndexes;
    int         currentBatchIndex;
};

#endif /* VOCPASCALINPUTLAYER_H */
