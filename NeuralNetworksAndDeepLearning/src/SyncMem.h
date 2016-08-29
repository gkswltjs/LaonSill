/*
 * SyncMem.h
 *
 *  Created on: 2016. 8. 24.
 *      Author: jhkim
 */

#ifndef SYNCMEM_H_
#define SYNCMEM_H_

#include <stddef.h>
#include <vector>

#include "Util.h"

/**
 * @brief 복사할 SRC와 DEST를 지정하는 열거형
 */
enum SyncMemCopyType {
	HostToHost=0,			// 호스트에서 호스트 복사
	HostToDevice=1,			// 호스트에서 디바이스 복사
	DeviceToHost=2,			// 디바이스에서 호스트 복사
	DeviceToDevice=3		// 디바이스에서 디바이스 복사
};


/**
 * @brief 동기화된 호스트, 디바이스 메모리 클래스
 * @details 읽기만 수행할 경우 non_mutable getter를 통해 포인터를 조회할 것
 *          mutable getter를 이용하는 경우 무조건 데이터에 변경이 있다고 보고
 *          대응 메모리 조회시 복사가 일어나게 된다.
 */
template <typename Dtype>
class SyncMem {
public:
	SyncMem();
	virtual ~SyncMem();

	/**
	 * @details 지정된 사이즈로 호스트와 디바이스에 메모리를 할당, 초기화한다.
	 * @param size 할당할 메모리 사이즈
	 */
	void reshape(size_t size);

	/**
	 * @details 읽기 전용의 호스트 메모리 포인터를 조회한다.
	 * @return 읽기 전용의 호스트 메모리 포인터
	 */
	const Dtype* host_mem();
	/**
	 * @details 읽기 전용의 디바이스 메모리 포인터를 조회한다.
	 * @return 읽기 전용의 디바이스 메모리 포인터
	 */
	const Dtype* device_mem();

	/**
	 * @details 읽기/쓰기 가능한 호스트 메모리 포인터를 조회한다.
	 * @return 읽기/쓰기 가능한 호스트 메모리 포인터
	 */
	Dtype* mutable_host_mem();
	/**
	 * @details 읽기/쓰기 가능한 디바이스 메모리 포인터를 조회한다.
	 * @return 읽기/쓰기 가능한 디바이스 메모리 포인터
	 */
	Dtype* mutable_device_mem();

	/**
	 * @details 주어진 데이터를 지정된 SRC에서 지정된 DEST로 복사한다.
	 * @param mem 복사할 데이터 메모리 포인터
	 * @param copyType 데이터를 복사할 SRC/DEST를 지정하는 열거형
	 */
	void set_mem(const Dtype* mem, SyncMemCopyType copyType);

	/**
	 * @details 호스트 메모리를 0으로 초기화한다.
	 */
	void reset_host_mem();
	/**
	 * @details 디바이스 메모리를 0으로 초기화한다.
	 */
	void reset_device_mem();

	/**
	 * @details 호스트 메모리에 주어진 호스트 포인터의 메모리값을 더한다.
	 * @param mem 더 할 호스트 메모리의 포인터
	 */
	void add_host_mem(const Dtype* mem);

	/**
	 * @details 디바이스 메모리에 주어진 디바이스 포인터의 메모리값을 더한다.
	 * @param mem 더 할 디바이스 메모리의 포인터
	 */
	void add_device_mem(const Dtype* mem);

	/**
	 * @details 호스트 메모리의 값을 스케일링한다.
	 * @param scale 스케일링할 스케일
	 */
	void scale_host_mem(const float scale);
	/**
	 * @details 디바이스 메모리의 값을 스케일링한다.
	 * @param scale 스케일링할 스케일
	 */
	void scale_device_mem(const float scale);

	/**
	 * @details 디바이스 메모리의 값을 제곱합한다.
	 * @return 디바이스 메모리 값의 제곱합
	 */
	double sumsq_device_mem();

	/**
	 * @details 메모리의 값을 출력한다.
	 * @param head 출력할 때 헤드에 쓰일 문구
	 */
	void print(const string& head);
	/**
	 * @details 메모리의 값을 출력한다.
	 * @param head 출력할 때 헤드에 쓰일 문구
	 * @param shape 출력 포맷 shape (batches, columns, rows, columns)
	 */
	void print(const string& head, const std::vector<uint32_t>& shape);


private:
	/**
	 * @details 디바이스 메모리의 변경 여부를 확인하고
	 *          변경된 경우 호스트 메모리를 업데이트한다.
	 * @param reset 변경 여부 플래그를 리셋할지 여부
	 */
	void checkDeviceMemAndUpdateHostMem(bool reset=true);
	/**
	 * @details 호스트 메모리의 변경 여부를 확인하고
	 *          변경된 경우 디바이스 메모리를 업데이트한다.
	 * @param reset 변경 여부 플래그를 리셋할지 여부
	 */
	void checkHostMemAndUpdateDeviceMem(bool reset=true);

	/**
	 * @details shape가 결정되어 메모리가 할당되었는지 확인
	 */
	void checkMemValidity();



private:
	size_t _size;							///< 할당된 메모리의 크기

	Dtype* _host_mem;						///< 호스트 메모리 포인터
	Dtype* _device_mem;						///< 디바이스 메모리 포인터

	bool _host_mem_updated;					///< 호스트 메모리 변경 여부 플래그
	bool _device_mem_updated;				///< 디바이스 메모리 변경 여부 플래그

};

#endif /* SYNCMEM_H_ */
