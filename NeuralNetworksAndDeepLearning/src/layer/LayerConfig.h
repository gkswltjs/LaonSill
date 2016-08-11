/**
 * @file LayerConfig.h
 * @date 2016/5/14
 * @author jhkim
 * @brief 레이어 설정과 관련된 열거형, 구조체 등을 선언
 * @details
 */

#ifndef LAYERCONFIG_H_
#define LAYERCONFIG_H_

#include <cmath>
#include <chrono>
#include <random>

#include "../Util.h"


//typedef arma::fvec rvec;
//typedef arma::fmat rmat;
//typedef arma::fcube rcube;
//typedef unsigned int UINT;


class Layer;




/**
 * @brief 학습 파라미터 초기화 타입 열거형
 * @details	지원하는 학습 파라미터 초기화 타입 열거,
 *          현재 Constant, Xavier, Gaussian을 지원.
 */
enum class ParamFillerType {
	None=0,				// 초기화하지 않음
	Constant=1,			// 특정값으로 초기화
	Xavier=2,			// 특정 범위의 uniform distribution을 따르는 값으로 초기화
	Gaussian=3			// Gaussian distribution을 따르는 값으로 초기화
};


/**
 * @brief 데이터의 구조정보를 정의하는 구조체
 */
struct io_dim {
    UINT rows;				///< 데이터의 행 수, 일반 1차원 데이터의 경우 한 데이터의 유효 엘리먼트의 수, 이미지와 같은 3차원 데이터의 경우 이미지의 height값을 따른다.
    UINT cols;				///< 데이터의 열 수, 일반 1차원 데이터의 경우 1, 이미지와 같은 3차원 데이터의 경우 이미지의 width값을 따른다.
    UINT channels;			///< 데이터의 채널 수, 일반 1차원 데이터의 경우 1, 이미지와 같은 3차원 데이터의 경우 RGB채널이 있는 경우 3, GrayScale인 경우 1의 값이 된다.
    UINT batches;			///< 한 번에 학습하는 데이터의 수를 의미한다.

    //io_dim(UINT rows=1, UINT cols=1, UINT channels=1, UINT batches=1) {
    io_dim() {

    }
    io_dim(UINT rows, UINT cols, UINT channels, UINT batches) {
    	this->rows = rows;
    	this->cols = cols;
    	this->channels = channels;
    	this->batches = batches;
    }
    //int size() const { return rows*cols*channels*batches; }

    /**
     * @details 데이터 하나의 엘리먼트 크기를 조회한다.
     * @return 데이터 하나의 엘리먼트 크기
     */
    int unitsize() const { return rows*cols*channels; }
    /**
     * @details 배치 단위의 데이터 엘리먼트 크기를 조회한다.
     * @return 배치 단위의 데이터 엘리먼트 크기
     */
    int batchsize() const { return rows*cols*channels*batches; }
};


/**
 * @brief 컨볼루션 연산을 정의하는 파라미터 구조체
 * @todo io_dim의 구조를 따르는 면이 있어(row, colum, channel...) 상속받았으나 batch값은 적합하지 않음.
 *       상속받지 않고 별도의 필드를 정의하는 것이 바람직해 보인다.
 */
struct filter_dim : public io_dim {
	UINT filters;			///< 컨볼루션 결과 출력 채널(필터)의 수
	UINT stride;			///< 컨볼루션을 적용할 stride 크기

	//filter_dim(UINT rows=1, UINT cols=1, UINT channels=1, UINT filters=1, UINT stride=1) : io_dim(rows, cols, channels) {
	filter_dim() {}
	filter_dim(UINT rows, UINT cols, UINT channels, UINT filters, UINT stride) : io_dim(rows, cols, channels, 1) {
		this->filters = filters;
		this->stride = stride;
	}
	/**
	 * @details 전체 필터의 엘리먼트 크기를 조회한다.
	 * @return 전체 필터의 엘리먼트 크기
	 */
	int size() const { return rows*cols*channels*filters; }
};

/**
 * @brief 풀링 연산을 정의하는 파라미터 구조체
 * @todo padding 적용을 위한 필드를 추가해야 함
 */
struct pool_dim {
	UINT rows;				///< 풀링 커널의 height 값
	UINT cols;				///< 풀링 커널의 width 값
	UINT stride;			///< 풀링을 적용할 stride 크기

	//pool_dim(UINT rows=1, UINT cols=1, UINT stride=1) {
	pool_dim() {}
	pool_dim(UINT rows, UINT cols, UINT stride) {
		this->rows = rows;
		this->cols = cols;
		this->stride = stride;
	}
};


/**
 * @brief LRN 연산을 정의하는 파라미터 구조체
 * @details 각 파라미터는 (1+(α/n)∑ixi^2)^β 수식을 참고한다.
 */
struct lrn_dim {
	UINT local_size;		///< 정규화를 적용할 채널의 수
	double alpha;			///< 스케일링 파라미터
	double beta;			///< 지수
	double k;

	//lrn_dim(UINT local_size=5, double alpha=1, double beta=5) {
	lrn_dim(UINT local_size=5, double alpha=0.0001, double beta=0.75, double k=2.0) {
		this->local_size = local_size;
		this->alpha = alpha;
		this->beta = beta;
		this->k = k;
	}
};

/**
 * @brief 학습 파라미터 업데이트 파라미터 구조체
 */
struct update_param {
	double lr_mult;			///< learning rate
	double decay_mult;		///< weight decay

	update_param() {}
	update_param(double lr_mult, double decay_mult) {
		this->lr_mult = lr_mult;
		this->decay_mult = decay_mult;
	}
};

/**
 * @brief 학습 파라미터 초기화 파라미터 구조체
 */
struct param_filler {
	ParamFillerType type;	///< 파라미터 초기화 타입
	double value;			///< 파라미터 초기화 관련 값

	param_filler() {}
	param_filler(ParamFillerType type, double value=0) {
		this->type = type;
		this->value = value;
	}

#ifndef GPU_MODE
	void fill(rvec &param, int n_in) {
		switch(type) {
		case ParamFillerType::Constant: param.fill(value); break;
		case ParamFillerType::Xavier:
			param.randn();
			param *= sqrt(3.0/n_in);
			break;
		case ParamFillerType::Gaussian:
			param.randn();
			break;
		case ParamFillerType::None:
		default:
			break;
		}
	}

	void fill(rmat &param, int n_in) {
		switch(type) {
		case ParamFillerType::Constant:
			param.fill(value);
			break;
		case ParamFillerType::Xavier:
			param.randn();
			param *= sqrt(3.0/n_in);				// initial point scaling
			break;
		case ParamFillerType::Gaussian:
			param.randn();
			break;
		case ParamFillerType::None:
		default:
			break;
		}
	}

	void fill(rcube &param, int n_in) {
		switch(type) {
		case ParamFillerType::Constant:
			param.fill(value);
			break;
		case ParamFillerType::Xavier:
			param.randn();
			param *= sqrt(3.0/n_in);				// initial point scaling
			break;
		case ParamFillerType::Gaussian:
			param.randn();
			break;
		case ParamFillerType::None:
		default:
			break;
		}
	}

#else
	/**
	 * @details 학습 파라미터를 초기화한다.
	 * @param param 파라미터 장치 메모리 포인터
	 * @param size 파라미터 사이즈
	 * @param n_in 레이어의 입력 노드 수
	 * @param n_out 레이어의 출력 노드 수
	 */
	void fill(DATATYPE *param, int size, int n_in, int n_out) {
		UINT i;
		switch(type) {
		case ParamFillerType::Constant:
		{
			//memset(param, value, size);
			for(int i = 0; i < size; i++) param[i] = value;
		}
			break;

		// ret = Nd4j.randn(order, shape).divi(FastMath.sqrt(shape[0] + shape[1]));
		// N(0, 1), {channel in, channel out, kernel x, kernel y},
		case ParamFillerType::Xavier:
		{
			//float sd_xavier = sqrt(1.0f / (n_in+n_out));
			//float sd_xavier = sqrt(3.0f / (n_out));

			float sd_xavier = sqrt(3.0f / (n_in));
			cout << "sd_xavier: " << sd_xavier << endl;
			std::random_device rd_xavier;
			std::mt19937 gen_xavier(rd_xavier());
			//std::uni _distribution<DATATYPE> normal_dist(0.0, 1.0);
			std::uniform_real_distribution<DATATYPE> unifrom_dist(-sd_xavier, sd_xavier);
			for(i = 0; i < size; i++) param[i] = unifrom_dist(gen_xavier);

			/*
			std::random_device rd_xavier;
			std::mt19937 gen_xavier(rd_xavier());
			std::normal_distribution<DATATYPE> normal_dist(0.0, 1.0);
			//float sd_xavier = sqrt(1.0f / (n_in+n_out));
			float sd_xavier = sqrt(3.0f / n_in);
			cout << "sd_xavier: " << sd_xavier << endl;
			for(i = 0; i < size; i++) param[i] = normal_dist(gen_xavier);//*sd_xavier;
			*/

		}
			break;
		case ParamFillerType::Gaussian:
		{
			float sd_gaussian = sqrt(1.0f/n_out);
			cout << "sd_gaussian: " << sd_gaussian << endl;
			std::random_device rd_gaussian;
			std::mt19937 gen_gaussian(rd_gaussian());
			std::normal_distribution<DATATYPE> normal_dist(0.0, sd_gaussian);
			for(i = 0; i < size; i++) param[i] = normal_dist(gen_gaussian)*sd_gaussian;
		}
			break;
		case ParamFillerType::None:
		default:
			break;
		}
	}
#endif

};



struct next_layer_relation {
	Layer *next_layer;
	UINT idx;

	next_layer_relation() {}
	next_layer_relation(Layer *next_layer, UINT idx) {
		this->next_layer = next_layer;
		this->idx = idx;
	}
};

struct prev_layer_relation {
	Layer *prev_layer;
	UINT idx;

	prev_layer_relation() {}
	prev_layer_relation(Layer *prev_layer, UINT idx) {
		this->prev_layer = prev_layer;
		this->idx = idx;
	}
};









#endif /* LAYERCONFIG_H_ */





























