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
		std::vector<uint32_t> _shape;
		std::string _source;
		std::string _sourceType;
        std::string _imageDir;
        bool        _cropImage;
        int         _croppedImageRow;
        int         _croppedImageCol;

		Builder() {
			this->type = Layer<Dtype>::CelebAInput;
            this->_imageDir = "";
            this->_cropImage = false;
            this->_croppedImageRow = -1;
            this->_croppedImageCol = -1;
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
        Builder* cropImage(int row, int col) {
            this->_cropImage = true;
            this->_croppedImageRow = row;
            this->_croppedImageCol = col;
        }
		Layer<Dtype>* build() {
			return new CelebAInputLayer(this);
		}
	};

    CelebAInputLayer();
    
	CelebAInputLayer(const std::string name, const std::string imageDir, bool cropImage,
        int croppedImageRow, int croppedImageCol);
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
	void initialize(const std::string imageDir, bool cropImage, int croppedImageRow,
        int croppedImageCol);

    std::string imageDir;
    bool        cropImage;
    int         imageCount;

    int         imageRow;
    int         imageCol;
    int         imageChannel;

    void        fillImagePaths();
    void        loadImages();
    void        loadPixels(cv::Mat image, int imageIndex);
    void        shuffleImages();

    std::vector<int> imageIndexes;
    std::vector<std::string> imagePaths;
    Dtype*      images;
};
#endif /* CELEBAINPUTLAYER_H */
