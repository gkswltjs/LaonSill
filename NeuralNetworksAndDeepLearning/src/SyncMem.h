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

class SyncMem {
public:
	enum CopyType {
		HostToHost=0,
		HostToDevice=1,
		DeviceToHost=2,
		DeviceToDevice=3
	};


	SyncMem();
	virtual ~SyncMem();

	void reshape(size_t size);

	const DATATYPE* host_mem();
	const DATATYPE* device_mem();

	DATATYPE* mutable_host_mem();
	DATATYPE* mutable_device_mem();


	void set_mem(const DATATYPE* mem, CopyType copyType);

	void reset_host_mem();
	void reset_device_mem();

	void add_host_mem(const DATATYPE* mem);
	void add_device_mem(const DATATYPE* mem);

	void scale_host_mem(const float scale);
	void scale_device_mem(const float scale);

	//DATATYPE sumsq_host_mem();
	DATATYPE sumsq_device_mem();


private:
	void checkDeviceMemAndUpdateHostMem();
	void checkHostMemAndUpdateDeviceMem();

	void checkMemValidity();



private:
	size_t _size;

	DATATYPE* _host_mem;
	DATATYPE* _device_mem;

	bool _host_mem_updated;
	bool _device_mem_updated;

};

#endif /* SYNCMEM_H_ */
