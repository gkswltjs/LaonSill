/*
 * SyncMem.h
 *
 *  Created on: 2016. 8. 24.
 *      Author: jhkim
 */

#ifndef SYNCMEM_H_
#define SYNCMEM_H_

#include <stddef.h>

#include "Util.h"

enum SyncMemCopyType {
	HostToHost=0,
	HostToDevice=1,
	DeviceToHost=2,
	DeviceToDevice=3
};



template <typename Dtype>
class SyncMem {
public:
	SyncMem();
	virtual ~SyncMem();

	void reshape(size_t size);

	const Dtype* host_mem();
	const Dtype* device_mem();

	Dtype* mutable_host_mem();
	Dtype* mutable_device_mem();


	void set_mem(const Dtype* mem, SyncMemCopyType copyType);

	void reset_host_mem();
	void reset_device_mem();

	void add_host_mem(const Dtype* mem);

	/**
	 * @details float, double 타입의 template에 한해 사용
	 */
	void add_device_mem(const Dtype* mem);

	void scale_host_mem(const float scale);
	/**
	 * @details float, double 타입의 template에 한해 사용
	 */
	void scale_device_mem(const float scale);

	//Dtype sumsq_host_mem();
	/**
	 * @details float, double 타입의 template에 한해 사용
	 */
	float sumsq_device_mem();


	void print(const string& head);


private:
	void checkDeviceMemAndUpdateHostMem();
	void checkHostMemAndUpdateDeviceMem();

	void checkMemValidity();



private:
	size_t _size;

	Dtype* _host_mem;
	Dtype* _device_mem;

	bool _host_mem_updated;
	bool _device_mem_updated;

};

#endif /* SYNCMEM_H_ */
