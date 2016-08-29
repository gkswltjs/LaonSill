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




template <typename Dtype>
class Data {
public:
	Data();
	virtual ~Data();

	void reshape(const vector<uint32_t>& shape);

	const Dtype* host_data();
	const Dtype* device_data();
	const Dtype* host_grad();
	const Dtype* device_grad();

	Dtype* mutable_host_data();
	Dtype* mutable_device_data();
	Dtype* mutable_host_grad();
	Dtype* mutable_device_grad();




	void set_host_data(Data* data, const uint32_t offset=0) { set_host_data(data->host_data()+offset); }
	void set_device_data(Data* data, const uint32_t offset=0) { set_device_data(data->device_data()+offset); }
	void set_host_grad(Data* grad, const uint32_t offset=0) { set_host_grad(grad->host_grad()+offset); }
	void set_device_grad(Data* grad, const uint32_t offset=0) { set_device_grad(grad->device_grad()+offset); }

	//void set_data(const Dtype* data, CopyType copyType);
	void set_host_data(const Dtype* data);
	void set_host_with_device_data(const Dtype* data);
	void set_device_with_host_data(const Dtype* data);
	void set_device_data(const Dtype* data);

	void set_host_grad(const Dtype* grad);
	void set_device_grad(const Dtype* grad);


	void reset_host_data();
	void reset_device_data();
	void reset_host_grad();
	void reset_device_grad();


	void add_host_data(Data* data, const uint32_t offset=0) { add_host_data(data->host_data()+offset); }
	void add_device_data(Data* data, const uint32_t offset=0) { add_device_data(data->device_data()+offset); }
	void add_host_grad(Data* grad, const uint32_t offset=0) { add_host_grad(grad->host_grad()+offset); }
	void add_device_grad(Data* grad, const uint32_t offset=0) { add_device_grad(grad->device_grad()+offset); }

	void add_host_data(const Dtype* data);
	void add_device_data(const Dtype* data);
	void add_host_grad(const Dtype* grad);
	void add_device_grad(const Dtype* grad);


	void scale_host_data(const float scale);
	void scale_device_data(const float scale);
	void scale_host_grad(const float scale);
	void scale_device_grad(const float scale);


	double sumsq_device_data();
	double sumsq_device_grad();


	inline uint32_t batches() const { return _shape[0]; }
	inline uint32_t channels() const { return _shape[1]; }
	inline uint32_t height() const { return _shape[2]; }
	inline uint32_t width() const { return _shape[3]; }


	void print_data(const string& head);
	void print_grad(const string& head);



private:
	vector<uint32_t> _shape;

	SyncMem<Dtype> _data;
	SyncMem<Dtype> _grad;

	size_t _count;

	const static uint32_t SHAPE_SIZE = 4;




public:
	static uint32_t printConfig;
};



#endif /* DATA_H_ */
