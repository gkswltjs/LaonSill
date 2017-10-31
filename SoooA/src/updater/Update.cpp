/**
 * @file Update.cpp
 * @date 2017-05-30
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "Update.h"
#include "PropMgmt.h"
#include "SysLog.h"
#include "Worker.h"
#include "Network.h"
#include "MathFunctions.h"
#include "StdOutLog.h"

template <typename Dtype>
int Update<Dtype>::currentStep = 0;

template<typename Dtype>
void Update<Dtype>::updateParam(UpdateContext context, Data<Dtype>* dataHistory,
    Data<Dtype>* dataHistory2, Data<Dtype>* data) {

    if (!SLPROP_BASE(updateGrad))
        return;

    int paramSize = context.paramSize;
    Dtype regScale = (Dtype)context.regScale;
    Dtype learnScale = (Dtype)context.learnScale;
    Dtype epsilon = (Dtype)context.epsilon;
    Dtype decayRate = (Dtype)context.decayRate;
    Dtype beta1 = (Dtype)context.beta1;
    Dtype beta2 = (Dtype)context.beta2;
    Dtype decayedBeta1 = (Dtype)context.decayedBeta1;
    Dtype decayedBeta2 = (Dtype)context.decayedBeta2;

	const uint32_t batches = SNPROP(batchSize);
	const Dtype normScale = 1.0/batches;
	const Dtype momentum = SNPROP(momentum);
	const Dtype negativeOne = -1.0;
    const Dtype negativeLearnScale = (-1.0) * learnScale;

#if 0
    if (!Worker<Dtype>::isSingle())
        data->mutable_host_grad();
#endif

	Dtype* d_paramGrad = data->mutable_device_grad();
	Dtype* d_paramData = data->mutable_device_data();
	Dtype* d_paramHistoryData = dataHistory->mutable_device_data();
	Dtype* d_paramHistoryData2 = dataHistory2->mutable_device_data();

    // apply optimizer
    Optimizer opt = (Optimizer)SNPROP(optimizer);
    if (opt == Optimizer::Momentum) {
        /****
         * Momentum Alogorithm
         *
         * v = mu * v - learning_rate * dx
         * x += v
         *
         */
    	soooa_gpu_axpy(paramSize, regScale, d_paramData, d_paramGrad);
		soooa_gpu_axpby(paramSize, learnScale, d_paramGrad, momentum,
				d_paramHistoryData);
		soooa_copy(paramSize, d_paramHistoryData, d_paramGrad);
		// update
		soooa_gpu_axpy(paramSize, negativeOne, d_paramGrad, d_paramData);
    } else if (opt == Optimizer::Vanilla) {
        /****
         * Vanilla Alogorithm
         *
         * x += -learning_rate * dx
         *
         */
    	checkCudaErrors(cublasSscal(Cuda::cublasHandle, paramSize,
            &learnScale, d_paramGrad, 1));				//
    	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, paramSize,
            &negativeOne, d_paramGrad, 1, d_paramData, 1));		// update
    } else if (opt == Optimizer::Nesterov) {
        /****
         * Nesterov Alogorithm
         *
         * v_prev = v # back this up
         * v = mu * v - learning_rate * dx # velocity update stays the same
         * x += -mu * v_prev + (1 + mu) * v # position update changes form
         *
         */
	    Update<Dtype>::doNesterov(paramSize, d_paramGrad,
            d_paramHistoryData, d_paramHistoryData2, d_paramData, momentum, learnScale);
    } else if (opt == Optimizer::Adagrad) {
        /****
         * Adagrad Alogorithm
         *
         * cache += dx**2
         * x += -learning_rate * dx / (sqrt(cache) + eps)
         *
         */
	    Update<Dtype>::doAdagrad(paramSize, d_paramGrad,
            d_paramHistoryData, d_paramData, learnScale, epsilon);

    } else if (opt == Optimizer::RMSprop) {
        /****
         * RMSprop
         *
         * cache = decay_rate * cache + (1 - decay_rate) * dx**2
         * x += - learning_rate * dx / (sqrt(cache) + eps)
         *
         */
	    Update<Dtype>::doRMSprop(paramSize, d_paramGrad,
            d_paramHistoryData, d_paramData, learnScale, epsilon, decayRate);

    } else if (opt == Optimizer::Adam) {
        /****
         * Adam
         *
         * m = beta1 * m + (1 - beta1) * dx
         * v = beta2 * v + (1 - beta2) * (dx**2)
         * x += -learning_rate * m / (sqrt(v) + eps)
         *
         */
        Update<Dtype>::doAdam(paramSize, d_paramGrad, d_paramHistoryData,
            d_paramHistoryData2, d_paramData, learnScale, epsilon, beta1, beta2,
            decayedBeta1, decayedBeta2);
    } else {
        SASSERT(false, "invalid optimizer. optimizer=%d", (int)opt);
    }
}

/**
 * learning rate policy (from CAFFE definition)
 *    - fixed: always return base_lr.
 *    - step: return base_lr * gamma ^ (floor(iter / step))
 *    - exp: return base_lr * gamma ^ iter
 *    - inv: return base_lr * (1 + gamma * iter) ^ (- power)
 *    - multistep: similar to step but it allows non uniform steps defined by
 *      stepvalue
 *    - poly: the effective learning rate follows a polynomial decay, to be
 *      zero by the max_iter. return base_lr (1 - iter/max_iter) ^ (power)
 *    - sigmoid: the effective learning rate follows a sigmod decay
 *      return base_lr ( 1/(1 + exp(-gamma * (iter - stepsize))))
 */
template <typename Dtype>
float Update<Dtype>::calcLearningRate() {
	float rate;
	switch (SNPROP(lrPolicy)) {
        case Fixed: {
            rate = SNPROP(baseLearningRate);
        }
            break;
        case Step: {
            currentStep = SNPROP(iterations) / SNPROP(stepSize);
            rate = SNPROP(baseLearningRate) * pow(SNPROP(gamma), currentStep);

            if (SNPROP(rate) != rate) {
            	STDOUT_LOG("Learning rate updated: : %f", rate);
            }
            if (SNPROP(rate) < 0.0f || SNPROP(rate) != rate) {
                SNPROP(rate) = rate;
            }
        }
            break;
        case Poly: {
            rate = SNPROP(baseLearningRate) * 
                pow(1.0 - (float)SNPROP(iterations) / 
                           ((float)SNPROP(epochs) * (float)SNPROP(miniBatch)), SNPROP(power));
        }
            break;
        case Multistep: {
        	const std::vector<int>& stepValue = SNPROP(stepValue);
        	const int iterations = SNPROP(iterations);
        	if (currentStep < stepValue.size() &&
        			iterations >= stepValue[currentStep]) {
        		currentStep++;
        		STDOUT_LOG("MultiStep Status: Iteration %d, step = %d", iterations,
        				currentStep);
        	}
        	rate = SNPROP(baseLearningRate) * pow(SNPROP(gamma), currentStep);
        }
        break;

        default: {
            exit(1);
        }
	}

	return rate;
}

template<typename Dtype>
UpdateContext Update<Dtype>::makeContext(int paramSize, float regScale, float learnScale) {
    UpdateContext context;
    context.paramSize = paramSize;
    context.regScale = regScale;
    context.learnScale = learnScale;
    context.epsilon = SNPROP(epsilon);
    context.decayRate = SNPROP(decayRate);
    context.beta1 = SNPROP(beta1);
    context.beta2 = SNPROP(beta2);
    context.decayedBeta1 = SLPROP_LEARN(decayedBeta1);
    context.decayedBeta2 = SLPROP_LEARN(decayedBeta2);

    return context;
}

template class Update<float>;
