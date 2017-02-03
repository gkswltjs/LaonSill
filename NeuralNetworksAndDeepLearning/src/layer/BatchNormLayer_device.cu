/**
 * @file BatchNormLayer_device.cu
 * @date 2017-01-25
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "cuda_runtime.h"

#include "BatchNormLayer.h"
#include "Exception.h"
#include "NetworkConfig.h"
#include "SysLog.h"
#include "StdOutLog.h"
#include "ColdLog.h"

#define BATCHCONDLAYER_LOG  0

using namespace std;

#ifdef GPU_MODE

template<typename Dtype>
BatchNormLayer<Dtype>::~BatchNormLayer() {
    if (this->depth == 0)
        return;

    SASSERT0(this->gammaSet != NULL);
    free(this->gammaSet);
    SASSERT0(this->betaSet != NULL);
    free(this->betaSet);
    SASSERT0(this->meanSet != NULL);
    free(this->meanSet);
    SASSERT0(this->varSet != NULL);
    free(this->varSet);
    SASSERT0(this->normInputSet != NULL);
    free(this->normInputSet);
}

template <typename Dtype>
void BatchNormLayer<Dtype>::update() {
    // FIXME: momentum, decay 등의 factor들을 고려하지 않았다.
    //        이런 부분을 고려하면 더 좋은 결과가 나올지도 모른다.
    float learningRate = this->networkConfig->getLearningRate();

    Dtype* gammaData = this->gammaSet->mutable_host_data();
    Dtype* gammaGrad = this->gammaSet->mutable_host_grad();
    Dtype* betaData = this->betaSet->mutable_host_data();
    Dtype* betaGrad = this->betaSet->mutable_host_grad();

    for (int i = 0; i < this->depth; i++) {
        gammaData[i] -= learningRate * gammaGrad[i];
        betaData[i]  -= learningRate * betaGrad[i];
    }
}

template <typename Dtype>
void BatchNormLayer<Dtype>::feedforward() {
    // FIXME: 현재 CPU 코드로 구현이 되어 있다. GPU 코드로 변경하자.
    // (1) mini-batch mean 값을 구한다.
	const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
	int batchCount = inputShape[0];

    const Dtype* inputData = this->_inputData[0]->host_data();
    Dtype* outputData = this->_outputData[0]->mutable_host_data();

	if (this->networkConfig->_status == NetworkStatus::Train) {
        Dtype* means = this->meanSet->mutable_host_data();
        Dtype* vars = this->varSet->mutable_host_data();

        // (1) mini-batch에 사용하는 mean, variance를 초기화 한다.
        for (int i = 0; i < this->depth; i++) {
            means[i] = 0;
            vars[i] = 0;
        }

        // (2) mini-batch mean 값을 구한다.
        for (int i = 0; i < this->depth; i++) {
            for (int j = 0; j < batchCount; j++) {
                int index = j * this->depth + i;
                means[i] += inputData[index];
            }
            means[i] = means[i] / (Dtype)batchCount;
        }

        // (3) mini-batch variance 값을 구한다.
        for (int i = 0; i < this->depth; i++) {
            for (int j = 0; j < batchCount; j++) {
                int index = j * this->depth + i;
                vars[i] += (inputData[index] - means[i]) * (inputData[index] - means[i]);
            }
            vars[i] = vars[i] / (Dtype)batchCount;
        }

        // (4) normalize 
        Dtype* normInputs = this->normInputSet->mutable_host_data();
        const Dtype* gammas = this->gammaSet->host_data();
        const Dtype* betas = this->betaSet->host_data();

        for (int i = 0; i < this->depth; i++) {
            Dtype denominator = sqrt(vars[i] + (Dtype)this->epsilon);
            for (int j = 0; j < batchCount; j++) {
                int index = j * this->depth + i;
                normInputs[index] = (inputData[index] - means[i]) / denominator;
                outputData[index] = normInputs[index] * gammas[i] + betas[i];
            }
        }

        // (5) global meanSets과 varianceSets를 갱신한다.
        this->batchSetCount += 1;

        Dtype* meanSums = this->meanSumSet->mutable_host_mem();
        Dtype* varSums = this->varSumSet->mutable_host_mem();
        for (int i = 0; i < this->depth; i++) {
            meanSums[i] += means[i];
            varSums[i] += vars[i];
        }
	} else if (this->networkConfig->_status == NetworkStatus::Test) {
        SASSERT((this->batchSetCount > 0), "need train before inference");
        STDOUT_LOG("Batch Norm Test Mode!!!");

        const Dtype* meanSums = this->meanSumSet->host_mem();
        const Dtype* varSums = this->varSumSet->host_mem();
        const Dtype* gammas = this->gammaSet->host_data();
        const Dtype* betas = this->betaSet->host_data();
        for (int i = 0; i < this->depth; i++) {
            Dtype avgMean = meanSums[i] / (Dtype)this->batchSetCount;
            Dtype avgVariance;
            if (this->batchSetCount == 1) {
                avgVariance = varSums[i];
            } else {
                avgVariance = varSums[i] / (Dtype)(this->batchSetCount - 1);
            }
            Dtype sqrtVariance = sqrt(avgVariance + this->epsilon);

            for (int j = 0; j < batchCount; j++) {
                int index = j * this->depth + i;
                outputData[index] = inputData[index] * gammas[i] / sqrtVariance +
                    betas[i] - gammas[i] * avgMean / sqrtVariance;
            }
        }
    } else {
        SASSERT(false, "Invalid network status =%d", this->networkConfig->_status);
    }
}

template <typename Dtype>
void BatchNormLayer<Dtype>::reshape() {
	if (!Layer<Dtype>::_adjustInputShape()) {
		const uint32_t count = Util::vecCountByAxis(this->_inputShape[0], 1);
		const uint32_t inputDataCount = this->_inputData[0]->getCountByAxis(1);
		assert(count == inputDataCount);
	}

	if (!Layer<Dtype>::_isInputShapeChanged(0))
		return;

	const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();

    // XXX: 현재 FC에 대해서만 생각하였음
    // TODO: Conv Layer에 대한 구현 필요
	uint32_t batches = inputShape[0];
	uint32_t channels = inputShape[1];
	uint32_t rows = inputShape[2];
	uint32_t cols = inputShape[3];
    uint32_t depth = this->_inputData[0]->getCountByAxis(1);

	this->_inputShape[0] = {batches, channels, rows, cols};
	this->_outputData[0]->reshape({batches, channels, rows, cols});

	STDOUT_COND_LOG(BATCHCONDLAYER_LOG, 
        "<%s> layer' input-0 has reshaped as: %dx%dx%dx%d\n",
        this->name.c_str(), batches, channels, rows, cols);
	STDOUT_COND_LOG(BATCHCONDLAYER_LOG,
	    "<%s> layer' output-0 has reshaped as: %dx%dx%dx%d\n", 
        this->name.c_str(), batches, channels, rows, cols);

    // Batch Normalization 과정에 필요한 구조체들의 메모리를 할당한다.
    if (this->depth == 0) {
        this->depth = depth;

        this->gammaSet->reshape({1, channels, rows, cols});
        this->betaSet->reshape({1, channels, rows, cols});
        this->meanSet->reshape({1, channels, rows, cols});
        this->varSet->reshape({1, channels, rows, cols});

        int allocSize = sizeof(Dtype) * this->depth;
        this->meanSumSet->reshape((size_t)allocSize);
        this->varSumSet->reshape((size_t)allocSize);

        this->normInputSet->reshape({batches, channels, rows, cols});

        // FIXME: 더 좋은 초기화 방법이 있을지도 모른다..
        Dtype* gammas = this->gammaSet->mutable_host_data();
        Dtype* betas = this->betaSet->mutable_host_data();
        Dtype* meanSums = this->meanSumSet->mutable_host_mem();
        Dtype* varSums = this->varSumSet->mutable_host_mem();
        for (int i = 0; i < this->depth; i++) {
            gammas[i] = 1;
            betas[i] = 0;
            meanSums[i] = 0;
            varSums[i] = 0;
        }
    } else {
        SASSERT0(this->depth == depth);
    }
}

template <typename Dtype>
void BatchNormLayer<Dtype>::computeNormInputGrad() {
    const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
    int batchCount = inputShape[0];

    const Dtype* outputGrad = this->_outputData[0]->host_grad();
    Dtype* normInputGrads = this->normInputSet->mutable_host_grad();
    const Dtype* gammas = this->gammaSet->host_data();

    for (int i = 0; i < this->depth; i++) {
        for (int j = 0; j < batchCount; j++) {
            int index = j * this->depth + i;
            normInputGrads[index] = outputGrad[index] * gammas[i];
        }
    }
}

template <typename Dtype>
void BatchNormLayer<Dtype>::computeVarianceGrad() {
	const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
	int batchCount = inputShape[0];
    const Dtype* inputData = this->_inputData[0]->host_data();
    Dtype* varGrads = this->varSet->mutable_host_grad();
    const Dtype* normInputGrads = this->normInputSet->host_grad();
    const Dtype* means = this->meanSet->host_data();
    const Dtype* vars = this->varSet->host_data();

    for (int i = 0; i < this->depth; i++) {
        varGrads[i] = 0;
        for (int j = 0; j < batchCount; j++) {
            int index = j * this->depth + i;
            varGrads[i] += normInputGrads[index] * (inputData[index] - means[i]) * (-0.5) *
                pow((vars[i] + this->epsilon), -1.5);
        }
    }
}

template <typename Dtype>
void BatchNormLayer<Dtype>::computeMeanGrad() {
	const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
	int batchCount = inputShape[0];
    const Dtype* inputData = this->_inputData[0]->host_data();
    Dtype* meanGrads = this->meanSet->mutable_host_grad();
    const Dtype* normInputGrads = this->normInputSet->host_grad();
    const Dtype* vars = this->varSet->host_data();
    const Dtype* varGrads = this->varSet->host_grad();
    const Dtype* means = this->meanSet->host_data();

    for (int i = 0; i < this->depth; i++) {
        meanGrads[i] = 0;
        for (int j = 0; j < batchCount; j++) {
            int index = j * this->depth + i;
            meanGrads[i] += normInputGrads[index] * (-1) / sqrt(vars[i] + this->epsilon) +
                varGrads[i] * (-2) * (inputData[index] - means[i]) / batchCount;
        }
    }
}

template <typename Dtype>
void BatchNormLayer<Dtype>::computeInputGrad() {
	const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
	int batchCount = inputShape[0];
    const Dtype* inputData = this->_inputData[0]->host_data();
	Dtype* inputGrad = this->_inputData[0]->mutable_host_grad();
    const Dtype* normInputGrads = this->normInputSet->host_grad();
    const Dtype* vars = this->varSet->host_data();
    const Dtype* varGrads = this->varSet->host_grad();
    const Dtype* means = this->meanSet->host_data();
    const Dtype* meanGrads = this->meanSet->host_grad();

    for (int i = 0; i < this->depth; i++) {
        for (int j = 0; j < batchCount; j++) {
            int index = j * this->depth + i;
            inputGrad[index] = normInputGrads[index] / sqrt(vars[i] + this->epsilon) +
                varGrads[i] * 2 * (inputData[index] - means[i]) / batchCount +
                meanGrads[i] / batchCount;
        }
    }
}

template <typename Dtype>
void BatchNormLayer<Dtype>::computeScaleGrad() {
	const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
	int batchCount = inputShape[0];
    const Dtype* outputGrad = this->_outputData[0]->host_grad();
    Dtype* gammaGrads = this->gammaSet->mutable_host_grad();
    const Dtype* normInputs = this->normInputSet->host_data();

    for (int i = 0; i < this->depth; i++) {
        gammaGrads[i] = 0;
        for (int j = 0; j < batchCount; j++) {
            int index = j * this->depth + i;
            gammaGrads[i] += outputGrad[index] * normInputs[index];
        }
    }
}

template <typename Dtype>
void BatchNormLayer<Dtype>::computeShiftGrad() {
	const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
	int batchCount = inputShape[0];
    const Dtype* outputGrad = this->_outputData[0]->host_grad();
    Dtype* betaGrads = this->betaSet->mutable_host_grad();

    for (int i = 0; i < this->depth; i++) {
        betaGrads[i] = 0;
        for (int j = 0; j < batchCount; j++) {
            int index = j * this->depth + i;
            betaGrads[i] += outputGrad[index];
        }
    }
}

template <typename Dtype>
void BatchNormLayer<Dtype>::backpropagation() {
    /*
     * 아래와 같은 simple한 network layer가 있다고 가정하자.
     *
     *               <<<< ith layer >>>>                        <<<< i+1th layer >>>>
     *   .....    Xi  Norm    ^Xi   γi * ^Xi + βi      Yi (=Xi+1)  ........
     *   .....    O ---------  O  ---------------------  O         ........
     *                                                     dL/dYi is already computed
     *
     *  (※  Xi = i번째 layer의 input 값, Norm = normaliztion
     *      ^Xi = i번째 layer의 중간 값, γi = scale factor, βi = shift factor
     *      Yi = i번째 layer의 ouput 값, i+1 번째 layer의 input 값이기도 함
     *      L = loss, dL/dYi = i+1번째 layer에서 계산되었던 gradient 값)
     *
     *  BatchNormLayer에서는 γi, βi를 학습해야 하는데 그것을 위해서 dL/dγi, dL/dβi를 계산해야
     *  한다. 또한, 하위 layer에 전달할 dL/dXi이 필요하다.
     *
     *  논문(https://arxiv.org/abs/1502.03167)에서 각각의 계산식이 있기 때문에 그것을 이용하여
     *  연산을 하도록 하자.)
     */

    // (1) dL/d^Xi = dL/dYi * γi
    computeNormInputGrad();

    // (2) dL/dSquaredSigma
    computeVarianceGrad();

    // (3) dL/dMean
    computeMeanGrad();

    // (4) dL/dXi
    computeInputGrad();

    // (5) dL/dγi
    computeScaleGrad();

    // (6) dL/dβi
    computeShiftGrad();
}

template BatchNormLayer<float>::~BatchNormLayer();
template void BatchNormLayer<float>::reshape();
template void BatchNormLayer<float>::update();
template void BatchNormLayer<float>::feedforward();
template void BatchNormLayer<float>::backpropagation();

#endif
