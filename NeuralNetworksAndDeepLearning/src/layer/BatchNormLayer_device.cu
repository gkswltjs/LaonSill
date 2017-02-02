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

    SASSERT0(this->gammaSets != NULL);
    free(this->gammaSets);
    SASSERT0(this->betaSets != NULL);
    free(this->betaSets);
    SASSERT0(this->meanSumSets != NULL);
    free(this->meanSumSets);
    SASSERT0(this->varianceSumSets != NULL);
    free(this->varianceSumSets);
    SASSERT0(this->localMeanSets != NULL);
    free(this->localMeanSets);
    SASSERT0(this->localVarianceSets != NULL);
    free(this->localVarianceSets);

    SASSERT0(this->normInputValues != NULL);
    free(this->normInputValues);
    SASSERT0(this->normInputGradValues != NULL);
    free(this->normInputGradValues);
    SASSERT0(this->varianceGradValues != NULL);
    free(this->varianceGradValues);
    SASSERT0(this->meanGradValues != NULL);
    free(this->meanGradValues);
    SASSERT0(this->gammaGradValues != NULL);
    free(this->gammaGradValues);
    SASSERT0(this->betaGradValues != NULL);
    free(this->betaGradValues);
}

template <typename Dtype>
void BatchNormLayer<Dtype>::update() {
    // FIXME: momentum, decay 등의 factor들을 고려하지 않았다.
    //        이런 부분을 고려하면 더 좋은 결과가 나올지도 모른다.
    float learningRate = this->networkConfig->getLearningRate();

    for (int i = 0; i < this->depth; i++) {
        this->gammaSets[i]  -= learningRate * this->gammaGradValues[i];
        this->betaSets[i]   -= learningRate * this->betaGradValues[i];

    }
}

template <typename Dtype>
void BatchNormLayer<Dtype>::feedforward() {
    // FIXME: 현재 CPU 코드로 구현이 되어 있다. GPU 코드로 변경하자.
    // (1) mini-batch mean 값을 구한다.
	const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
	int batchCount = inputShape[0];

    SASSUME0(this->localMeanSets != NULL);
    SASSUME0(this->localVarianceSets != NULL);

    const Dtype* inputData = this->_inputData[0]->host_data();
    Dtype* outputData = this->_outputData[0]->mutable_host_data();

    // (1) mini-batch에 사용하는 localMeanSets, localVarainceSets를 초기화 한다.
    for (int i = 0; i < this->depth; i++) {
        this->localMeanSets[i] = 0;
        this->localVarianceSets[i] = 0;
    }

    // (2) mini-batch mean 값을 구한다.
    for (int i = 0; i < this->depth; i++) {
        for (int j = 0; j < batchCount; j++) {
            int index = j * this->depth + i;
            this->localMeanSets[i] += inputData[index];
        }
    }

    for (int i = 0; i < this->depth; i++) {
        this->localMeanSets[i] = this->localMeanSets[i] / (Dtype)batchCount;
    }

    // (3) mini-batch variance 값을 구한다.
    for (int i = 0; i < this->depth; i++) {
        for (int j = 0; j < batchCount; j++) {
            int index = j * this->depth + i;
            this->localVarianceSets[i] += 
                (inputData[index] - this->localMeanSets[i]) *
                (inputData[index] - this->localMeanSets[i]);
        }
    }

    for (int i = 0; i < this->depth; i++) {
        this->localVarianceSets[i] = this->localVarianceSets[i] / (Dtype)batchCount;
    }

    // (4) normalize 
    for (int i = 0; i < this->depth; i++) {
        Dtype denominator = sqrt(this->localVarianceSets[i] + (Dtype)this->epsilon);
        for (int j = 0; j < batchCount; j++) {
            int index = j * this->depth + i;
            this->normInputValues[index] = 
                (inputData[index] - this->localMeanSets[i]) / denominator;
            outputData[index] = 
                this->normInputValues[index] * this->gammaSets[i] + this->betaSets[i];
        }
    }

    // (5) global meanSets과 varianceSets를 갱신한다.
    this->batchSetCount += 1;

    for (int i = 0; i < this->depth; i++) {
        this->meanSumSets[i] += this->localMeanSets[i];
        this->varianceSumSets[i] = this->localVarianceSets[i];
    }

    for (int i = 0; i < 10; i++) {
        COLD_LOG(ColdLog::INFO, true, "BN[-]: %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n", 
            inputData[0], inputData[1], inputData[2], inputData[3], inputData[4],
            inputData[5], inputData[6], inputData[7], inputData[8], inputData[9]);
        COLD_LOG(ColdLog::INFO, true, "BN[+]: %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n", 
            outputData[0], outputData[1], outputData[2], outputData[3], outputData[4],
            outputData[5], outputData[6], outputData[7], outputData[8], outputData[9]);
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
	uint32_t channels = 1;
	uint32_t rows = this->_inputData[0]->getCountByAxis(1);
	uint32_t cols = 1;

	this->_inputShape[0] = {batches, channels, rows, cols};
	this->_outputData[0]->reshape({batches, channels, rows, cols});

	STDOUT_COND_LOG(BATCHCONDLAYER_LOG, 
        "<%s> layer' input-0 has reshaped as: %dx%dx%dx%d\n",
        this->name.c_str(), batches, channels, rows, cols);
	STDOUT_COND_LOG(BATCHCONDLAYER_LOG,
	    "<%s> layer' output-0 has reshaped as: %dx%dx%dx%d\n", 
        this->name.c_str(), batches, channels, rows, cols);

    if (this->depth == 0)
        this->depth = rows;
    else
        SASSERT0(this->depth == rows);

    // Batch Normalization 과정에 필요한 구조체들의 메모리를 할당한다.
    if (this->gammaSets == NULL) {
        SASSERT0(this->betaSets == NULL);
        SASSERT0(this->meanSumSets == NULL);
        SASSERT0(this->varianceSumSets == NULL);
        SASSERT0(this->localMeanSets == NULL);
        SASSERT0(this->localVarianceSets == NULL);

        SASSERT0(this->normInputValues == NULL);
        SASSERT0(this->meanGradValues == NULL);
        SASSERT0(this->varianceGradValues == NULL);
        SASSERT0(this->gammaGradValues == NULL);
        SASSERT0(this->betaGradValues == NULL);

        int allocSize = sizeof(Dtype) * this->depth;

        this->gammaSets = (Dtype*)malloc(allocSize);
        SASSERT0(this->gammaSets != NULL);
        this->betaSets = (Dtype*)malloc(allocSize);
        SASSERT0(this->betaSets != NULL);
        this->meanSumSets = (Dtype*)malloc(allocSize);
        SASSERT0(this->meanSumSets != NULL);
        this->varianceSumSets = (Dtype*)malloc(allocSize);
        SASSERT0(this->varianceSumSets != NULL);
        this->localMeanSets = (Dtype*)malloc(allocSize);
        SASSERT0(this->localMeanSets != NULL);
        this->localVarianceSets = (Dtype*)malloc(allocSize);
        SASSERT0(this->localVarianceSets != NULL);

        this->varianceGradValues = (Dtype*)malloc(allocSize);
        SASSERT0(this->varianceGradValues != NULL);
        this->meanGradValues = (Dtype*)malloc(allocSize);
        SASSERT0(this->meanGradValues != NULL);
        this->gammaGradValues = (Dtype*)malloc(allocSize);
        SASSERT0(this->gammaGradValues != NULL);
        this->betaGradValues = (Dtype*)malloc(allocSize);
        SASSERT0(this->betaGradValues != NULL);

        // XXX: batches가 크게 변하지 않을꺼라는 가정이 있다. 
        int batchAllocSize = allocSize * batches;
        this->normInputValues = (Dtype*)malloc(batchAllocSize);
        SASSERT0(this->normInputValues != NULL);
        this->normInputGradValues = (Dtype*)malloc(batchAllocSize);
        SASSERT0(this->normInputGradValues != NULL);

        for (int i = 0; i < this->depth; i++) {
            this->gammaSets[i] = 1;
            this->betaSets[i] = 0;
            this->meanSumSets[i] = 0;
            this->varianceSumSets[i] = 0;
        }
    } else {
        SASSERT0(this->betaSets != NULL);
        SASSERT0(this->meanSumSets != NULL);
        SASSERT0(this->varianceSumSets != NULL);
        SASSERT0(this->localMeanSets != NULL);
        SASSERT0(this->localVarianceSets != NULL);

        SASSERT0(this->varianceGradValues != NULL);
        SASSERT0(this->meanGradValues != NULL);
        SASSERT0(this->gammaGradValues != NULL);
        SASSERT0(this->betaGradValues != NULL);

        SASSERT0(this->normInputValues != NULL);
        SASSERT0(this->normInputGradValues != NULL);
    }
}

template <typename Dtype>
void BatchNormLayer<Dtype>::computeNormInputGrad() {
	const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
	int batchCount = inputShape[0];

    const Dtype* outputGrad = this->_outputData[0]->host_grad();

    for (int i = 0; i < this->depth; i++) {
        for (int j = 0; j < batchCount; j++) {
            int index = j * this->depth + i;
            this->normInputGradValues[index] = outputGrad[index] * this->gammaSets[i];
        }
    }
}

template <typename Dtype>
void BatchNormLayer<Dtype>::computeVarianceGrad() {
	const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
	int batchCount = inputShape[0];
    const Dtype* inputData = this->_inputData[0]->host_data();

    for (int i = 0; i < this->depth; i++) {
        this->varianceGradValues[i] = 0;
        for (int j = 0; j < batchCount; j++) {
            int index = j * this->depth + i;
            this->varianceGradValues[i] += 
                this->normInputGradValues[index] * 
                (inputData[index] - this->localMeanSets[i]) * (-0.5) *
                pow((this->localVarianceSets[i] + this->epsilon), -1.5);
        }
    }
}

template <typename Dtype>
void BatchNormLayer<Dtype>::computeMeanGrad() {
	const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
	int batchCount = inputShape[0];
    const Dtype* inputData = this->_inputData[0]->host_data();

    for (int i = 0; i < this->depth; i++) {
        this->meanGradValues[i] = 0;
        for (int j = 0; j < batchCount; j++) {
            int index = j * this->depth + i;
            this->meanGradValues[i] +=
                this->normInputGradValues[index] * (-1) /
                sqrt(this->localVarianceSets[i] + this->epsilon) +
                this->varianceGradValues[i] * (-2) * 
                (inputData[index] - this->localMeanSets[i]) / batchCount;
        }
    }
}

template <typename Dtype>
void BatchNormLayer<Dtype>::computeInputGrad() {
	const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
	int batchCount = inputShape[0];
    const Dtype* inputData = this->_inputData[0]->host_data();
	Dtype* inputGrad = this->_inputData[0]->mutable_host_grad();

    for (int i = 0; i < this->depth; i++) {
        for (int j = 0; j < batchCount; j++) {
            int index = j * this->depth + i;
            inputGrad[index] = this->normInputGradValues[index] /
                sqrt(this->localVarianceSets[i] + this->epsilon) +
                this->varianceGradValues[i] * 2 *
                (inputData[index] - this->localMeanSets[i]) / batchCount +
                this->meanGradValues[i] / batchCount;
        }
    }
}

template <typename Dtype>
void BatchNormLayer<Dtype>::computeScaleGrad() {
	const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
	int batchCount = inputShape[0];
    const Dtype* outputGrad = this->_outputData[0]->host_grad();

    for (int i = 0; i < this->depth; i++) {
        this->gammaGradValues[i] = 0;
        for (int j = 0; j < batchCount; j++) {
            int index = j * this->depth + i;
            this->gammaGradValues[i] += outputGrad[index] * this->normInputValues[index];
        }
    }
}

template <typename Dtype>
void BatchNormLayer<Dtype>::computeShiftGrad() {
	const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
	int batchCount = inputShape[0];
    const Dtype* outputGrad = this->_outputData[0]->host_grad();

    for (int i = 0; i < this->depth; i++) {
        this->betaGradValues[i] = 0;
        for (int j = 0; j < batchCount; j++) {
            int index = j * this->depth + i;
            this->betaGradValues[i] += outputGrad[index];
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
