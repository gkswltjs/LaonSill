/*
 * DataReader.cpp
 *
 *  Created on: Jun 30, 2017
 *      Author: jkim
 */

#include "DataReader.h"
#include "Datum.h"

using namespace std;

template <typename T>
DataReader<T>::DataReader(const string& source)
: db(source, Mode::READ) {
	this->db.open();

	string value = this->db.getNextValue();
	this->numData = atoi(value.c_str());
}

template <typename T>
DataReader<T>::~DataReader() {
	this->db.close();

	while (!this->data_queue.empty()) {
		T* t = this->data_queue.front();
		this->data_queue.pop();
		if (t) {
			delete t;
		}
	}
}

template <>
Datum* DataReader<Datum>::getNextData() {
	if (this->data_queue.empty()) {
		string value = this->db.getNextValue();
		Datum* datum = new Datum();
		Datum::deserializeFromString(value, datum);
		return datum;
	} else {
		Datum* datum = this->data_queue.front();
		this->data_queue.pop();
		return datum;
	}
}

template <>
Datum* DataReader<Datum>::peekNextData() {
	if (this->data_queue.empty()) {
		string value = this->db.getNextValue();
		Datum* datum = new Datum();
		Datum::deserializeFromString(value, datum);
		this->data_queue.push(datum);
		return datum;
	} else {
		return this->data_queue.front();
	}
}


template <typename T>
int DataReader<T>::getNumData() {
	return this->numData;
}

template class DataReader<Datum>;

