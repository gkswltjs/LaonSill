/**
 * @file KistiInputLayer.h
 * @date 2017-03-28
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef KISTIINPUTLAYER_H
#define KISTIINPUTLAYER_H 

#include <string>

#include <opencv2/opencv.hpp>

#include "common.h"
#include "InputLayer.h"
#include "Layer.h"

#define KISTIDATA_DEFAULT_RESIZED_ROW_SIZE       224
#define KISTIDATA_DEFAULT_RESIZED_COL_SIZE       224

typedef struct KistiData_s {
    std::string filePath;
    std::vector<int> labels;
} KistiData;

template<typename Dtype>
class KistiInputLayer : public InputLayer<Dtype> {
public: 

	class Builder : public InputLayer<Dtype>::Builder {
	public:
		std::vector<uint32_t> _shape;
		std::string _source;
		std::string _sourceType;

        std::string _imageDir;
        int         _resizedImageRow;
        int         _resizedImageCol;
        bool        _train;

		Builder() {
			this->type = Layer<Dtype>::KistiInput;
            this->_imageDir = "";
            this->_resizedImageRow = KISTIDATA_DEFAULT_RESIZED_ROW_SIZE;
            this->_resizedImageCol = KISTIDATA_DEFAULT_RESIZED_COL_SIZE;
            this->_train = true;
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
        Builder* train(bool train) {
            this->_train = train;
            return this;
        }
        Builder* resize(int row, int col) {
            this->_resizedImageRow = row;
            this->_resizedImageCol = col;
            return this;
        }
		Layer<Dtype>* build() {
			return new KistiInputLayer(this);
		}
	};

    KistiInputLayer();
    
	KistiInputLayer(const std::string name, const std::string imageDir,
        int resizedImageRow, int resizedImageCol, bool train);
	KistiInputLayer(const std::string& name);
	KistiInputLayer(Builder* builder);

    virtual ~KistiInputLayer();

	void feedforward();
	using Layer<Dtype>::feedforward;
	void feedforward(const uint32_t baseIndex, const char* end=0);

    int getNumTrainData();
    int getNumTestData();
    void shuffleTrainDataSet();

	void reshape();
    void setTrain(bool train) { this->train = train; }

protected:
	void initialize(const std::string imageDir, int resizedImageRow, int resizedImageCol,
        bool train);

    std::string imageDir;
    int imageRow;
    int imageCol;
    int imageChannel;
    bool train;

public:
    std::map<std::string, int> keywordMap;
    std::vector<KistiData> trainData;
    std::vector<KistiData> testData;
protected:

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
#endif /* KISTIINPUTLAYER_H */
