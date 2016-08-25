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





class Data {
public:
	Data();
	virtual ~Data();

	void reshape(const vector<uint32_t>& shape);

	const DATATYPE* host_data();
	const DATATYPE* device_data();
	const DATATYPE* host_grad();
	const DATATYPE* device_grad();

	DATATYPE* mutable_host_data();
	DATATYPE* mutable_device_data();
	DATATYPE* mutable_host_grad();
	DATATYPE* mutable_device_grad();




	void set_host_data(Data* data, const uint32_t offset=0) { set_host_data(data->host_data()+offset); }
	void set_device_data(Data* data, const uint32_t offset=0) { set_device_data(data->device_data()+offset); }
	void set_host_grad(Data* grad, const uint32_t offset=0) { set_host_grad(grad->host_grad()+offset); }
	void set_device_grad(Data* grad, const uint32_t offset=0) { set_device_grad(grad->device_grad()+offset); }

	//void set_data(const DATATYPE* data, CopyType copyType);
	void set_host_data(const DATATYPE* data);
	void set_host_with_device_data(const DATATYPE* data);
	void set_device_with_host_data(const DATATYPE* data);
	void set_device_data(const DATATYPE* data);

	void set_host_grad(const DATATYPE* grad);
	void set_device_grad(const DATATYPE* grad);


	void reset_host_data();
	void reset_device_data();
	void reset_host_grad();
	void reset_device_grad();


	void add_host_data(Data* data, const uint32_t offset=0) { add_host_data(data->host_data()+offset); }
	void add_device_data(Data* data, const uint32_t offset=0) { add_device_data(data->device_data()+offset); }
	void add_host_grad(Data* grad, const uint32_t offset=0) { add_host_grad(grad->host_grad()+offset); }
	void add_device_grad(Data* grad, const uint32_t offset=0) { add_device_grad(grad->device_grad()+offset); }

	void add_host_data(const DATATYPE* data);
	void add_device_data(const DATATYPE* data);
	void add_host_grad(const DATATYPE* grad);
	void add_device_grad(const DATATYPE* grad);


	void scale_host_data(const float scale);
	void scale_device_data(const float scale);
	void scale_host_grad(const float scale);
	void scale_device_grad(const float scale);


	DATATYPE sumsq_device_data();
	DATATYPE sumsq_device_grad();


	inline uint32_t batches() const { return _shape[0]; }
	inline uint32_t channels() const { return _shape[1]; }
	inline uint32_t height() const { return _shape[2]; }
	inline uint32_t width() const { return _shape[3]; }


	void print_data(const string& head);
	void print_grad(const string& head);


private:
	void print(const DATATYPE* data, const string& head);


private:
	vector<uint32_t> _shape;
	SyncMem _data;
	SyncMem _grad;

	size_t _count;

	const static uint32_t SHAPE_SIZE = 4;




public:
	static uint32_t printConfig;
};

/*
class Data {
public:
	Data();
	virtual ~Data();

	enum CopyType {
		HostToHost=0,
		HostToDevice=1,
		DeviceToHost=2,
		DeviceToDevice=3
	};

	void reshape(const vector<uint32_t>& shape);

	const DATATYPE* cpu_data();
	const DATATYPE* gpu_data();
	const DATATYPE* cpu_grad();
	const DATATYPE* gpu_grad();

	DATATYPE* mutable_cpu_data();
	DATATYPE* mutable_gpu_data();
	DATATYPE* mutable_cpu_grad();
	DATATYPE* mutable_gpu_grad();


	void set_data(const DATATYPE* data, CopyType copyType);

	void set_cpu_data(Data* data, const uint32_t offset=0) { set_cpu_data(data->cpu_data()+offset); }
	void set_gpu_data(Data* data, const uint32_t offset=0) { set_gpu_data(data->gpu_data()+offset); }
	void set_cpu_grad(Data* grad, const uint32_t offset=0) { set_cpu_grad(grad->cpu_grad()+offset); }
	void set_gpu_grad(Data* grad, const uint32_t offset=0) { set_gpu_grad(grad->gpu_grad()+offset); }

	void set_cpu_data(const DATATYPE* data);
	void set_gpu_data(const DATATYPE* data);
	void set_cpu_grad(const DATATYPE* grad);
	void set_gpu_grad(const DATATYPE* grad);


	void reset_cpu_data();
	void reset_gpu_data();
	void reset_cpu_grad();
	void reset_gpu_grad();


	void add_cpu_data(Data* data, const uint32_t offset=0) { add_cpu_data(data->cpu_data()+offset); }
	void add_gpu_data(Data* data, const uint32_t offset=0) { add_gpu_data(data->gpu_data()+offset); }
	void add_cpu_grad(Data* grad, const uint32_t offset=0) { add_cpu_grad(grad->cpu_grad()+offset); }
	void add_gpu_grad(Data* grad, const uint32_t offset=0) { add_gpu_grad(grad->gpu_grad()+offset); }

	void add_cpu_data(const DATATYPE* data);
	void add_gpu_data(const DATATYPE* data);
	void add_cpu_grad(const DATATYPE* grad);
	void add_gpu_grad(const DATATYPE* grad);


	void scale_cpu_data(const float scale);
	void scale_gpu_data(const float scale);
	void scale_cpu_grad(const float scale);
	void scale_gpu_grad(const float scale);


	DATATYPE sumsq_gpu_data();
	DATATYPE sumsq_gpu_grad();


	inline uint32_t batches() const { return _shape[0]; }
	inline uint32_t channels() const { return _shape[1]; }
	inline uint32_t height() const { return _shape[2]; }
	inline uint32_t width() const { return _shape[3]; }


	void print_data(const string& head);
	void print_grad(const string& head);


private:
	bool alloc_cpu_data();
	bool alloc_gpu_data();
	bool alloc_cpu_grad();
	bool alloc_gpu_grad();

	void checkGpuDataAndUpdateCpuData();
	void checkCpuDataAndUpdateGpuData();
	void checkGpuGradAndUpdateCpuGrad();
	void checkCpuGradAndUpdateGpuGrad();

	void print(const DATATYPE* data, const string& head);




private:
	vector<uint32_t> _shape;

	DATATYPE* _cpu_data;
	DATATYPE* _gpu_data;
	DATATYPE* _cpu_grad;
	DATATYPE* _gpu_grad;

	bool _cpu_data_modified;
	bool _gpu_data_modified;
	bool _cpu_grad_modified;
	bool _gpu_grad_modified;

	size_t _count;

	const static uint32_t SHAPE_SIZE = 4;

public:
	static uint32_t printConfig;
};
*/

#endif /* DATA_H_ */
