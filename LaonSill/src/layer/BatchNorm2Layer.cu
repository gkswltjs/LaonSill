/*
 * BatchNorm2Layer.cpp
 *
 *  Created on: Dec 18, 2017
 *      Author: jkim
 */

#include "BatchNorm2Layer.h"
#include "SysLog.h"
#include "PropMgmt.h"
#include "MemoryMgmt.h"
#include "MathFunctions.h"
#include "Updater.h"

using namespace std;

template <typename Dtype>
BatchNorm2Layer<Dtype>::BatchNorm2Layer()
: LearnableLayer<Dtype>() {
	this->type = Layer<Dtype>::BatchNorm2;

	this->movingAverageFraction = SLPROP(BatchNorm2, movingAverageFraction);
	this->clipVariance = false;
	this->useGlobalStats = SLPROP(BatchNorm2, useGlobalStats);
	this->eps = std::max<float>(SLPROP(BatchNorm2, eps), 0.00001f);
	this->scaleBias = SLPROP(BatchNorm2, scaleBias);
	//if (param.has_scale_filler() ... )

	if (this->scaleBias) {
		this->_params.resize(5);
		this->_paramsHistory.resize(5);
		this->_paramsHistory2.resize(5);
		this->_paramsInitialized.resize(5);
	} else {
		this->_params.resize(3);
		this->_paramsHistory.resize(3);
		this->_paramsHistory2.resize(3);
		this->_paramsInitialized.resize(3);
	}

	LearnableLayer<Dtype>::initParam(0, "mean");
	LearnableLayer<Dtype>::initParam(1, "variance");
	LearnableLayer<Dtype>::initParam(2, "variance_correlation");

	if (this->scaleBias) {
		LearnableLayer<Dtype>::initParam(3, "scale");
		LearnableLayer<Dtype>::initParam(4, "bias");
	}
	this->iter = 0;

	// Mask statistics from optimization by setting local learning rates
	// for mean, variance, and the this->varcorrection to zero.
	this->updatePolicies.resize(3);
	for (int i = 0; i < 3; i++) {
		// set lr and decay = 0 for global mean and variance
		this->updatePolicies[i].lr_mult = 0.f;
		this->updatePolicies[i].decay_mult = 0.f;
	}
	// set lr for scale and bias to 1
	if (this->scaleBias) {
		this->updatePolicies.resize(5);
		for (int i = 3; i < 5; i++) {
			// set lr and decay = 1 for scale and bias
			this->updatePolicies[i].lr_mult = 1.f;
			this->updatePolicies[i].decay_mult = 1.f;
		}
	}

	SNEW(this->mean, Data<Dtype>, "mean");
	SASSUME0(this->mean != NULL);

	SNEW(this->var, Data<Dtype>, "var");
	SASSUME0(this->var != NULL);

	SNEW(this->invVar, Data<Dtype>, "invVar");
	SASSUME0(this->invVar != NULL);

	SNEW(this->onesC, Data<Dtype>, "onesC");
	SASSUME0(this->onesC != NULL);

	SNEW(this->onesN, Data<Dtype>, "onesN");
	SASSUME0(this->onesN != NULL);

	SNEW(this->onesHW, Data<Dtype>, "onesHW");
	SASSUME0(this->onesHW != NULL);

	SNEW(this->tempC, Data<Dtype>, "tempC");
	SASSUME0(this->tempC != NULL);

	SNEW(this->tempNC, Data<Dtype>, "tempNC");
	SASSUME0(this->tempNC != NULL);

	SNEW(this->tempNCHW, Data<Dtype>, "tempNCHW");
	SASSUME0(this->tempNCHW != NULL);

	SNEW(this->xNorm, Data<Dtype>, "xNorm");
	SASSUME0(this->xNorm != NULL);
}

template <typename Dtype>
BatchNorm2Layer<Dtype>::~BatchNorm2Layer() {
	for (int i = 0; i < this->_params.size(); i++) {
		SDELETE(this->_params[i]);
		SDELETE(this->_paramsHistory[i]);
		SDELETE(this->_paramsHistory2[i]);
	}

	SDELETE(this->mean);
	SDELETE(this->var);
	SDELETE(this->invVar);
	SDELETE(this->xNorm);
	SDELETE(this->onesN);
	SDELETE(this->onesHW);
	SDELETE(this->onesC);
	SDELETE(this->tempC);
	SDELETE(this->tempNC);
	SDELETE(this->tempNCHW);

	this->updateParams.clear();
}

template <typename Dtype>
void BatchNorm2Layer<Dtype>::reshape() {
	bool adjusted = Layer<Dtype>::_adjustInputShape();
	if (adjusted) {
		// XXX
		if (this->_inputData[0]->numAxes() == 1) {
			this->channels = 1;
		} else {
			this->channels = this->_inputData[0]->getShape(1);
		}

		LearnableLayer<Dtype>::reshapeParam(0, {1, 1, 1, (uint32_t)this->channels});
		LearnableLayer<Dtype>::reshapeParam(1, {1, 1, 1, (uint32_t)this->channels});
		LearnableLayer<Dtype>::reshapeParam(2, {1, 1, 1, 1});

		this->_params[0]->reset_host_data();
		this->_params[1]->reset_host_data();
		this->_params[2]->reset_host_data();

		if (this->scaleBias) {
			LearnableLayer<Dtype>::reshapeParam(3, {1, 1, 1, (uint32_t)this->channels});
			LearnableLayer<Dtype>::reshapeParam(4, {1, 1, 1, (uint32_t)this->channels});

			param_filler<Dtype>& scaleFiller = SLPROP(BatchNorm2, scaleFiller);
			scaleFiller.fill(this->_params[3]);

			param_filler<Dtype>& biasFiller = SLPROP(BatchNorm2, biasFiller);
			biasFiller.fill(this->_params[4]);
		}
	}
	if (!Layer<Dtype>::_isInputShapeChanged(0))
		return;

	// XXX
	if (this->_inputData[0]->numAxes() > 1) {
		SASSERT0(this->_inputData[0]->getShape(1) == this->channels);
	}
	this->_outputData[0]->reshapeLike(this->_inputData[0]);

	uint32_t N = this->_inputData[0]->getShape(0);
	uint32_t C = this->_inputData[0]->getShape(1);
	uint32_t H = this->_inputData[0]->getShape(2);
	uint32_t W = this->_inputData[0]->getShape(3);

	const vector<uint32_t> cShape = {1, 1, 1, (uint32_t)C};

	this->mean->reshape(cShape);
	this->var->reshape(cShape);
	this->invVar->reshape(cShape);

	this->onesN->reshape({1, 1, 1, N});
	this->onesC->reshape(cShape);
	this->onesHW->reshape({1, 1, 1, H * W});

	this->tempC->reshape(cShape);
	this->tempNC->reshape({1, 1, 1, N * C});
	this->tempNCHW->reshapeLike(this->_inputData[0]);

	this->xNorm->reshapeLike(this->_inputData[0]);

	this->onesN->reset_host_data(false, 1.f);
	this->onesC->reset_host_data(false, 1.f);
	this->onesHW->reset_host_data(false, 1.f);

	//this->tempC->reset_host_data();
	//this->tempNC->reset_host_data(false, 1.f);

}




/****************************************************************************
 *
 ****************************************************************************/

// multicast x[c] into y[.,c,...]
template <typename Dtype>
void BatchNorm2Layer<Dtype>::multicast_gpu(int N, int C, int S, const Dtype *x, Dtype *y ) {
  soooa_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N, C, 1, Dtype(1.),
     this->onesN->device_data(), x, Dtype(0.),
     this->tempNC->mutable_device_data());
  soooa_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N*C, S , 1, Dtype(1.),
     this->tempNC->device_data(), this->onesHW->device_data(), Dtype(0.), y);
}

// y[c] = sum x(.,c,...)
template <typename Dtype>
void BatchNorm2Layer<Dtype>::compute_sum_per_channel_gpu(int N, int C, int S, const Dtype *x, Dtype *y) {
  soooa_gpu_gemv<Dtype>(CblasNoTrans, N * C, S, Dtype(1.), x,
      this->onesHW->device_data(),
      Dtype(0.), this->tempNC->mutable_device_data());
  soooa_gpu_gemv<Dtype>(CblasTrans, N, C, Dtype(1.), this->tempNC->device_data(),
      this->onesN->device_data(), Dtype(0.), y);
}

// y[c] = mean x(.,c,...)
template <typename Dtype>
void BatchNorm2Layer<Dtype>::compute_mean_per_channel_gpu(int N, int C, int S, const Dtype *x, Dtype *y) {
  Dtype F = 1. / (N * S);
  compute_sum_per_channel_gpu(N, C, S, x, y);
  soooa_gpu_scal(C, F, y);
}












template <typename Dtype>
void BatchNorm2Layer<Dtype>::feedforward() {
	int N = this->_inputData[0]->getShape(0);
	int C = this->channels;
	int S = this->_inputData[0]->getCountByAxis(0) / (N * C);
	int outputSize = this->_outputData[0]->getCount();

	const Dtype* inputData = this->_inputData[0]->device_data();
	Dtype* outputData = this->_outputData[0]->mutable_device_data();
	const Dtype* globalMean = this->_params[0]->device_data();
	const Dtype* globalVar = this->_params[1]->device_data();

	if (SNPROP(status) == NetworkStatus::Test) {
		//  Y = X- EX
		multicast_gpu(N, C, S, globalMean, this->tempNCHW->mutable_device_data());
		soooa_gpu_sub(outputSize, inputData, this->tempNCHW->device_data(), outputData);
		//  inv_var = (eps + var)^(-0.5)
		soooa_copy(C, globalVar, this->var->mutable_device_data());
		soooa_gpu_add_scalar(C, Dtype(this->eps), this->var->mutable_device_data());
		soooa_gpu_powx(C, this->var->device_data(), Dtype(-0.5F),
				this->invVar->mutable_device_data());
		//  X_norm = (X-EX) * inv_var
		multicast_gpu(N, C, S, this->invVar->device_data(),
				this->tempNCHW->mutable_device_data());
		soooa_gpu_mul(outputSize, outputData, this->tempNCHW->device_data(), outputData);
	} else {  // if (this->phase_ == TRAIN)
		// temp = EX
		compute_mean_per_channel_gpu(N, C, S, inputData,
				this->mean->mutable_device_data());
		multicast_gpu(N, C, S, this->mean->device_data(),
				this->tempNCHW->mutable_device_data());
		// Y = X-EX
		soooa_gpu_sub(outputSize, inputData, this->tempNCHW->device_data(), outputData);
		// temp = (X-EX)^2;
		soooa_gpu_square(outputSize, this->_outputData[0]->device_data(),
				this->tempNCHW->mutable_device_data());
		compute_mean_per_channel_gpu(N, C, S, this->tempNCHW->device_data(),
				this->var->mutable_device_data());

		soooa_copy(C, this->var->device_data(),
				this->tempC->mutable_device_data());
		//  temp= 1/sqrt(e + var(c)
		soooa_gpu_add_scalar(C, Dtype(this->eps), this->tempC->mutable_device_data());
		soooa_gpu_powx(C, this->tempC->device_data(), Dtype(-0.5F),
				this->invVar->mutable_device_data());
		multicast_gpu(N, C, S, this->invVar->device_data(),
				this->tempNCHW->mutable_device_data());
		// X_norm = (X-mean(c)) / sqrt(e + var(c))
		soooa_gpu_mul(outputSize, outputData, this->tempNCHW->device_data(), outputData);
		// copy x_norm for backward
		soooa_copy(outputSize, outputData, this->xNorm->mutable_device_data());

	    //  update global mean and variance
	    if (this->iter > 1) {
	      soooa_gpu_axpby<Dtype>(C, Dtype(1. - this->movingAverageFraction),
	          this->mean->device_data(), Dtype(this->movingAverageFraction),
	          this->_params[0]->mutable_device_data());
	      soooa_gpu_axpby<Dtype>(C, Dtype((1. - this->movingAverageFraction)),
	          this->var->device_data(), Dtype(this->movingAverageFraction),
	          this->_params[1]->mutable_device_data());
	    } else {
	      soooa_copy<Dtype>(C, this->mean->device_data(),
	          this->_params[0]->mutable_device_data());
	      soooa_copy<Dtype>(C, this->var->device_data(),
	          this->_params[1]->mutable_device_data());
	    }
	    this->iter++;
	}

	//  -- STAGE 2:  Y = X_norm * scale[c] + shift[c]  -----------------
	if (this->scaleBias) {
		//  Y = X_norm * scale[c]
		multicast_gpu(N, C, S, this->_params[3]->device_data(),
				this->tempNCHW->mutable_device_data());
		soooa_gpu_mul(outputSize, outputData, this->tempNCHW->device_data(), outputData);
		//  Y = Y + shift[c]
		multicast_gpu(N, C, S, this->_params[4]->device_data(),
				this->tempNCHW->mutable_device_data());
		soooa_gpu_add(outputSize, outputData, this->tempNCHW->mutable_device_data(),
				outputData);
	}
}

template <typename Dtype>
void BatchNorm2Layer<Dtype>::backpropagation() {
	int N = this->_inputData[0]->getShape(0);
	int C = this->channels;
	int S = this->_inputData[0]->getCountByAxis(0) / (N * C);
	int outputSize = this->_outputData[0]->getCount();

	const Dtype* outputGrad = this->_outputData[0]->device_grad();
	//  --  STAGE 1: compute dE/d(scale) and dE/d(shift) ---------------
	if (this->scaleBias) {
		//  scaleGrad: dE/d(scale)  =  sum(dE/dY .* X_norm)
		Dtype* scaleGrad = this->_params[3]->mutable_device_grad();
		soooa_gpu_mul<Dtype>(outputSize, outputGrad, this->xNorm->device_data(),
		this->tempNCHW->mutable_device_grad());
		compute_sum_per_channel_gpu(N, C, S, this->tempNCHW->device_grad(), scaleGrad);
		//  shiftGrad: dE/d(shift) = sum (dE/dY)
		Dtype* shiftGrad = this->_params[4]->mutable_device_grad();
		compute_sum_per_channel_gpu(N, C, S, outputGrad, shiftGrad);

		// --  STAGE 2: backprop dE/d(x_norm) = dE/dY .* scale ------------
		//  dE/d(X_norm) = dE/dY * scale[c]
		const Dtype* scaleData = this->_params[3]->device_data();
		multicast_gpu(N, C, S, scaleData, this->tempNCHW->mutable_device_data());
		soooa_gpu_mul<Dtype>(outputSize, outputGrad, this->tempNCHW->device_data(),
		this->xNorm->mutable_device_grad());

		outputGrad = this->xNorm->device_grad();
	}
	// --  STAGE 3: backprop dE/dY --> dE/dX --------------------------

	// ATTENTION: from now on we will use notation Y:= X_norm
	const Dtype* outputData = this->xNorm->device_data();
	Dtype* inputGrad = this->_inputData[0]->mutable_device_grad();

	//  temp = mean(dE/dY .* Y)
	soooa_gpu_mul<Dtype>(outputSize, outputGrad, outputData,
	this->tempNCHW->mutable_device_grad());
	compute_mean_per_channel_gpu(N, C, S, this->tempNCHW->device_grad(),
	this->tempC->mutable_device_grad());
	multicast_gpu(N, C, S, this->tempC->device_grad(), this->tempNCHW->mutable_device_grad());

	// bottom = mean(dE/dY .* Y) .* Y
	soooa_gpu_mul<Dtype>(outputSize, this->tempNCHW->device_grad(), outputData, inputGrad);

	// temp = mean(dE/dY)
	compute_mean_per_channel_gpu(N, C, S, outputGrad,
	this->tempC->mutable_device_grad());
	multicast_gpu(N, C, S, this->tempC->device_grad(), this->tempNCHW->mutable_device_grad());

	// bottom = mean(dE/dY) + mean(dE/dY .* Y) .* Y
	soooa_gpu_add<Dtype>(outputSize, this->tempNCHW->device_grad(), inputGrad, inputGrad);

	// bottom = dE/dY - mean(dE/dY)-mean(dE/dY \cdot Y) \cdot Y
	soooa_gpu_sub<Dtype>(outputSize, outputGrad, inputGrad, inputGrad);

	// dE/dX = dE/dX ./ sqrt(var(X) + eps)
	multicast_gpu(N, C, S, this->invVar->device_data(),
	this->tempNCHW->mutable_device_data());
	soooa_gpu_mul<Dtype>(outputSize, inputGrad, this->tempNCHW->device_data(), inputGrad);
}

template <typename Dtype>
void BatchNorm2Layer<Dtype>::update() {
	//const uint32_t size = this->depth;
	const Dtype weightDecay = SNPROP(weightDecay);
	const Dtype learningRate = Update<float>::calcLearningRate();
	const Dtype beta1 = SNPROP(beta1);
	const Dtype beta2 = SNPROP(beta2);

	SLPROP(BatchNorm2, decayedBeta1) *= beta1;
	SLPROP(BatchNorm2, decayedBeta2) *= beta2;

	if (this->scaleBias) {
		SASSUME0(this->updateParams.size() == 5);
	} else {
		SASSUME0(this->updateParams.size() == 3);
	}

	for (int i = 0; i < 5; i++) {
		if (i >= 3 && !this->scaleBias) {
			continue;
		}
		int paramSize = this->_params[i]->getCount();
		Dtype regScale = weightDecay * this->updatePolicies[i].decay_mult;
		Dtype learnScale = learningRate * this->updatePolicies[i].lr_mult;
		UpdateContext context = Update<Dtype>::makeContext(paramSize, regScale, learnScale);
		this->updateParams[i].context = context;
	}

	Updater::updateParams(this->updateParams);
}

template <typename Dtype>
void BatchNorm2Layer<Dtype>::applyChanges(LearnableLayer<Dtype> *targetLayer) {
	SASSERT(false, "Not implemented yet.");
}

template <typename Dtype>
void BatchNorm2Layer<Dtype>::syncParams(LearnableLayer<Dtype> *targetLayer) {
	SASSERT(false, "Not implemented yet.");
}

























/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* BatchNorm2Layer<Dtype>::initLayer() {
	BatchNorm2Layer* layer = NULL;
	SNEW(layer, BatchNorm2Layer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void BatchNorm2Layer<Dtype>::destroyLayer(void* instancePtr) {
    BatchNorm2Layer<Dtype>* layer = (BatchNorm2Layer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void BatchNorm2Layer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
    SASSERT0(index == 0);

    BatchNorm2Layer<Dtype>* layer = (BatchNorm2Layer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == 0);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == 0);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool BatchNorm2Layer<Dtype>::allocLayerTensors(void* instancePtr) {
    BatchNorm2Layer<Dtype>* layer = (BatchNorm2Layer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void BatchNorm2Layer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
    BatchNorm2Layer<Dtype>* layer = (BatchNorm2Layer<Dtype>*)instancePtr;
    layer->feedforward();
}

template<typename Dtype>
void BatchNorm2Layer<Dtype>::backwardTensor(void* instancePtr) {
    BatchNorm2Layer<Dtype>* layer = (BatchNorm2Layer<Dtype>*)instancePtr;
    layer->backpropagation();
}

template<typename Dtype>
void BatchNorm2Layer<Dtype>::learnTensor(void* instancePtr) {
    BatchNorm2Layer<Dtype>* layer = (BatchNorm2Layer<Dtype>*)instancePtr;
    layer->update();
}

template class BatchNorm2Layer<float>;
