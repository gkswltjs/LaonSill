/**
 * @file EtriInputLayer.h
 * @date 2017-03-28
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef ETRIINPUTLAYER_H
#define ETRIINPUTLAYER_H 

#include <string>

#include <opencv2/opencv.hpp>

#include "common.h"
#include "InputLayer.h"
#include "Layer.h"

#define ETRIDATA_DEFAULT_RESIZED_ROW_SIZE       224
#define ETRIDATA_DEFAULT_RESIZED_COL_SIZE       224

typedef struct EtriData_s {
    std::string filePath;
    std::vector<int> labels;
} EtriData;

template<typename Dtype>
class EtriInputLayer : public InputLayer<Dtype> {
public: 

	class Builder : public InputLayer<Dtype>::Builder {
	public:
		std::vector<uint32_t> _shape;
		std::string _source;
		std::string _sourceType;

        std::string _imageDir;
        int         _resizedImageRow;
        int         _resizedImageCol;

		Builder() {
			this->type = Layer<Dtype>::EtriInput;
            this->_imageDir = "";
            this->_resizedImageRow = ETRIDATA_DEFAULT_RESIZED_ROW_SIZE;
            this->_resizedImageCol = ETRIDATA_DEFAULT_RESIZED_COL_SIZE;
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
        Builder* image(std::string imageDir) {
            this->_imageDir = imageDir;
            return this;
        }
        Builder* resize(int row, int col) {
            this->_resizedImageRow = row;
            this->_resizedImageCol = col;
            return this;
        }
		Layer<Dtype>* build() {
			return new EtriInputLayer(this);
		}
	};

    EtriInputLayer();
    
	EtriInputLayer(const std::string name, const std::string imageDir,
        int resizedImageRow, int resizedImageCol);
	EtriInputLayer(Builder* builder);

    virtual ~EtriInputLayer();

	void feedforward();
	using Layer<Dtype>::feedforward;
	void feedforward(const uint32_t baseIndex, const char* end=0);

    int getNumTrainData();
    int getNumTestData();
    void shuffleTrainDataSet();

	void reshape();

protected:
	void initialize(const std::string imageDir, int resizedImageRow, int resizedImageCol);

    std::string imageDir;
    int imageRow;
    int imageCol;
    int imageChannel;

    std::map<std::string, int> keywordMap;
    std::vector<EtriData> trainData;
    std::vector<EtriData> testData;

    void registerData(std::string filePath);
    void prepareKeywordMap();
    void prepareData();
    void loadPixels(cv::Mat image, int imageIndex);
    void loadImages(int batchIndex);
    void loadLabels(int batchIndex);
    void shuffleImages();

    Dtype *images;
    Dtype *labels;
    int currentBatchIndex;
};
#endif /* ETRIINPUTLAYER_H */
