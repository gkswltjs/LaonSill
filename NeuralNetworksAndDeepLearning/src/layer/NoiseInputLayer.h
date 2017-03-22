/**
 * @file NoiseInputLayer.h
 * @date 2017-02-16
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef NOISEINPUTLAYER_H
#define NOISEINPUTLAYER_H 

#include <string>

#include "common.h"
#include "InputLayer.h"
#include "Layer.h"

template <typename Dtype>
class NoiseInputLayer : public InputLayer<Dtype> {
public: 
	class Builder : public InputLayer<Dtype>::Builder {
	public:
		std::vector<uint32_t> _shape;
		std::string _source;
		std::string _sourceType;
        double _noiseRangeLow;
        double _noiseRangeHigh;
        int _noiseDepth;

        // linear transform
        bool _useLinearTrans;
        int _tranChannels;
        int _tranRows;
        int _tranCols;
        double _tranMean;
        double _tranVariance;

        bool _regenerateNoise;

		Builder() {
			this->type = Layer<Dtype>::NoiseInput;
            this->_noiseDepth = 100;
            this->_noiseRangeLow = -1.0;
            this->_noiseRangeHigh = 1.0;

            this->_useLinearTrans = false;
            this->_tranChannels = 1;
            this->_tranRows = 1;
            this->_tranCols = 1;
            this->_tranMean = 0;
            this->_tranVariance = 1.0;
            this->_regenerateNoise = true;
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
        Builder* noise(int depth, double rangeLow, double rangeHigh) {
            this->_noiseDepth = depth;
            this->_noiseRangeLow = rangeLow;
            this->_noiseRangeHigh = rangeHigh;
            return this;
        }
        Builder* regenerateNoise(bool regenerate) {
            this->_regenerateNoise = regenerate;
            return this;
        }
        Builder* linear(int channels, int rows, int cols, double mean, double variance) {
            this->_useLinearTrans = true;
            this->_tranChannels = channels;
            this->_tranRows = rows;
            this->_tranCols = cols;
            this->_tranMean = mean;
            this->_tranVariance = variance;
            return this;
        }
		Layer<Dtype>* build() {
			return new NoiseInputLayer(this);
		}
	};

    NoiseInputLayer();
	NoiseInputLayer(const std::string name, int noiseDepth, double noiseRangeLow,
        double noiseRangeHigh, bool useLinearTrans, int tranChannels, int tranRows,
        int tranCols, double tranMean, double tranVariance, bool regenerateNoise);
	NoiseInputLayer(Builder* builder);

    virtual ~NoiseInputLayer();

	void feedforward();
	using Layer<Dtype>::feedforward;
	void feedforward(const uint32_t baseIndex, const char* end=0);

    int getNumTrainData();
    int getNumTestData();
    void shuffleTrainDataSet();

	void reshape();

    void setRegenerateNoise(bool regenerate);

protected:
	void initialize(int noiseDepth, double noiseRangeLow, double noiseRangeHigh,
        bool useLinearTrans, int tranChannels, int tranRows, int tranCols, double tranMean,
        double tranVariance, bool regenerateNoise);
            
    bool prepareUniformArray();
    void prepareLinearTranMatrix();

    int noiseDepth;
    double noiseRangeLow;
    double noiseRangeHigh;

    bool useLinearTrans;
    int tranChannels;
    int tranRows;
    int tranCols;
    double tranMean;
    double tranVariance;

    bool regenerateNoise;

    int batchSize;

    Dtype* uniformArray;
    Dtype* linearTransMatrix;
};

#endif /* NOISEINPUTLAYER_H */
