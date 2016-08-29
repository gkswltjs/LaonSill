/*
 * Data.h
 *
 *  Created on: 2016. 8. 19.
 *      Author: jhkim
 */

#ifndef DATA_H_
#define DATA_H_

#include <cstdint>
#include <vector>

#include "Util.h"
#include "SyncMem.h"

using namespace std;



/**
 * @brief Layer 특정 단계에서의 data, gradient를 pair로 warpping, util method를 제공하는 클래스
 * @details Layer의 입력, 출력, Parameter에 해당하는 data와 gradient에 적용
 */
template <typename Dtype>
class Data {
public:
	Data();
	virtual ~Data();

	void reshape(const vector<uint32_t>& shape);

	/**
	 * @details 데이터의 수정할 수 없는 호스트 메모리 포인터 조회한다.
	 * @return 데이터의 수정할 수 없는 호스트 메모리 포인터
	 */
	const Dtype* host_data();
	/**
	 * @details 데이터의 수정할 수 없는 디바이스 메모리 포인터 조회한다.
	 * @return 데이터의 수정할 수 없는 디바이스 메모리 포인터
	 */
	const Dtype* device_data();
	/**
	 * @details 그레디언트의 수정할 수 없는 호스트 메모리 포인터 조회한다.
	 * @return 그레디언트의 수정할 수 없는 호스트 메모리 포인터
	 */
	const Dtype* host_grad();
	/**
	 * @details 그레디언트의 수정할 수 없는 디바이스 메모리 포인터 조회한다.
	 * @return 그레디언트의 수정할 수 없는 디바이스 메모리 포인터
	 */
	const Dtype* device_grad();

	/**
	 * @details 데이터의 수정할 수 있는 호스트 메모리 포인터 조회한다.
	 * @return 데이터의 수정할 수 있는 호스트 메모리 포인터
	 */
	Dtype* mutable_host_data();
	/**
	 * @details 데이터의 수정할 수 있는 디바이스 메모리 포인터 조회한다.
	 * @return 데이터의 수정할 수 있는 디바이스 메모리 포인터
	 */
	Dtype* mutable_device_data();
	/**
	 * @details 그레디언트의 수정할 수 있는 호스트 메모리 포인터 조회한다.
	 * @return 그레디언트의 수정할 수 있는 호스트 메모리 포인터
	 */
	Dtype* mutable_host_grad();
	/**
	 * @details 그레디언트의 수정할 수 있는 디바이스 메모리 포인터 조회한다.
	 * @return 그레디언트의 수정할 수 있는 디바이스 메모리 포인터
	 */
	Dtype* mutable_device_grad();

	/**
	 * @details 데이터의 호스트 메모리에 _count만큼 값을 복사한다.
	 * @param data 복사할 Data
	 * @param offset data의 포인터에 대한 offset
	 */
	void set_host_data(Data* data, const uint32_t offset=0) { set_host_data(data->host_data()+offset); }
	/**
	 * @details 데이터의 디바이스 메모리에 _count만큼 값을 복사한다.
	 * @param data 복사할 Data
	 * @param offset data의 포인터에 대한 offset
	 */
	void set_device_data(Data* data, const uint32_t offset=0) { set_device_data(data->device_data()+offset); }
	/**
	 * @details 그레디언트의 호스트 메모리에 _count만큼 값을 복사한다.
	 * @param grad 복사할 Data
	 * @param offset grad의 포인터에 대한 offset
	 */
	void set_host_grad(Data* grad, const uint32_t offset=0) { set_host_grad(grad->host_grad()+offset); }
	/**
	 * @details 그레디언트의 디바이스 메모리에 _count만큼 값을 복사한다.
	 * @param grad 복사할 Data
	 * @param offset grad의 포인터에 대한 offset
	 */
	void set_device_grad(Data* grad, const uint32_t offset=0) { set_device_grad(grad->device_grad()+offset); }

	/**
	 * @details 데이터의 호스트 메모리에 _count만큼 값을 복사한다.
	 * @param data 복사할 로우 데이터의 포인터
	 */
	void set_host_data(const Dtype* data);
	/**
	 * @details 데이터의 호스트 메모리에 _count만큼 주어진 디바이스 메모리 포인터로부터 값을 복사한다.
	 * @param data 복사할 로우 디바이스 데이터의 포인터
	 */
	void set_host_with_device_data(const Dtype* data);
	/**
	 * @details 데이터의 디바이스 메모리에 _count만큼 주어진 호스트 메모리 포인터로부터 값을 복사한다.
	 * @param data 복사할 로우 호스트 데이터의 포인터
	 */
	void set_device_with_host_data(const Dtype* data);
	/**
	 * @details 데이터의 디바이스 메모리에 _count만큼 값을 복사한다.
	 * @param data 복사할 로우 디바이스 데이터의 포인터
	 */
	void set_device_data(const Dtype* data);


	/**
	 * @details 그레디언트의 호스트 메모리에 _count만큼 값을 복사한다.
	 * @param data 복사할 로우 호스트 데이터의 포인터
	 */
	void set_host_grad(const Dtype* grad);
	/**
	 * @details 그레디언트의 디바이스 메모리에 _count만큼 값을 복사한다.
	 * @param data 복사할 로우 디바이스 데이터의 포인터
	 */
	void set_device_grad(const Dtype* grad);


	/**
	 * @details 데이터의 호스트 메모리를 0으로 초기화한다.
	 */
	void reset_host_data();
	/**
	 * @details 데이터의 디바이스 메모리를 0으로 초기화한다.
	 */
	void reset_device_data();
	/**
	 * @details 그레디언트의 호스트 메모리를 0으로 초기화한다.
	 */
	void reset_host_grad();
	/**
	 * @details 그레디언트의 디바이스 메모리를 0으로 초기화한다.
	 */
	void reset_device_grad();


	/**
	 * @details 데이터의 호스트 메모리에 주어진 Data의 값을 더한다.
	 * @param data 더 할 Data
	 * @param offset data의 포인터에 대한 offset
	 */
	void add_host_data(Data* data, const uint32_t offset=0) { add_host_data(data->host_data()+offset); }
	/**
	 * @details 데이터의 디바이스 메모리에 주어진 Data의 값을 더한다.
	 * @param data 더 할 Data
	 * @param offset data의 포인터에 대한 offset
	 */
	void add_device_data(Data* data, const uint32_t offset=0) { add_device_data(data->device_data()+offset); }
	/**
	 * @details 그레디언트의 호스트 메모리에 주어진 Data의 값을 더한다.
	 * @param grad 더 할 Data
	 * @param offset data의 포인터에 대한 offset
	 */
	void add_host_grad(Data* grad, const uint32_t offset=0) { add_host_grad(grad->host_grad()+offset); }
	/**
	 * @details 그레디언트의 디바이스 메모리에 주어진 Data의 값을 더한다.
	 * @param grad 더 할 Data
	 * @param offset data의 포인터에 대한 offset
	 */
	void add_device_grad(Data* grad, const uint32_t offset=0) { add_device_grad(grad->device_grad()+offset); }

	/**
	 * @details 데이터의 호스트 메모리에 주어진 로우 호스트 포인터의 메모리 값을 더한다.
	 * @param data 더 할 로우 호스트 메모리 포인터
	 */
	void add_host_data(const Dtype* data);
	/**
	 * @details 데이터의 디바이스 메모리에 주어진 로우 디바이스 포인터의 메모리 값을 더한다.
	 * @param data 더 할 로우 디바이스 메모리 포인터
	 */
	void add_device_data(const Dtype* data);
	/**
	 * @details 그레디언트의 호스트 메모리에 주어진 로우 호스트 포인터의 메모리 값을 더한다.
	 * @param grad 더 할 로우 호스트 메모리 포인터
	 */
	void add_host_grad(const Dtype* grad);
	/**
	 * @details 그레디언트의 디바이스 메모리에 주어진 로우 디바이스 포인터의 메모리 값을 더한다.
	 * @param grad 더 할 로우 디바이스 메모리 포인터
	 */
	void add_device_grad(const Dtype* grad);

	/**
	 * @details 데이터의 호스트 메모리를 스케일링한다.
	 * @param scale 스케일링할 스케일
	 */
	void scale_host_data(const float scale);
	/**
	 * @details 데이터의 디바이스 메모리를 스케일링한다.
	 * @param scale 스케일링할 스케일
	 */
	void scale_device_data(const float scale);
	/**
	 * @details 그레디언트의 호스트 메모리를 스케일링한다.
	 * @param scale 스케일링할 스케일
	 */
	void scale_host_grad(const float scale);
	/**
	 * @details 그레디언트의 디바이스 메모리를 스케일링한다.
	 * @param scale 스케일링할 스케일
	 */
	void scale_device_grad(const float scale);


	/**
	 * @details 데이터의 디바이스 메모리의 제곱합을 구한다.
	 * @param 데이터의 디바이스 메모리 제곱합
	 */
	double sumsq_device_data();
	/**
	 * @details 그레디언트의 디바이스 메모리의 제곱합을 구한다.
	 * @param 그레디언트의 디바이스 메모리 제곱합
	 */
	double sumsq_device_grad();


	/**
	 * @details Data의 batch shape를 조회한다.
	 * @return Data의 batch shape
	 */
	inline uint32_t batches() const { return _shape[0]; }
	/**
	 * @details Data의 channel shape를 조회한다.
	 * @return Data의 channel shape
	 */
	inline uint32_t channels() const { return _shape[1]; }
	/**
	 * @details Data의 height shape를 조회한다.
	 * @return Data의 height shape
	 */
	inline uint32_t height() const { return _shape[2]; }
	/**
	 * @details Data의 width shape를 조회한다.
	 * @return Data의 width shape
	 */
	inline uint32_t width() const { return _shape[3]; }

	/**
	 * @details 데이터를 shape에 따라 화면에 출력한다.
	 * @param head 출력할 때 헤드에 쓰일 문구
	 */
	void print_data(const string& head);
	/**
	 * @details 그레디언트를 shape에 따라 화면에 출력한다.
	 * @param head 출력할 때 헤드에 쓰일 문구
	 */
	void print_grad(const string& head);



private:
	vector<uint32_t> _shape;			///< Data의 shape, Batches, Channels, Rows, Columns의 4차원 벡터로 구성

	SyncMem<Dtype> _data;				///< Data의 데이터
	SyncMem<Dtype> _grad;				///< Data의 그레디언트

	size_t _count;						///< Data 메모리의 크기, 엘레먼트의 수 (Batches*Channels*Rows*Columns)

	const static uint32_t SHAPE_SIZE = 4;

public:
	static uint32_t printConfig;
};



#endif /* DATA_H_ */
