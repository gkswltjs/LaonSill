/*
 * SDF.cpp
 *
 *  Created on: Jun 28, 2017
 *      Author: jkim
 */

#include <sys/stat.h>

#include "iostream"
#include "SDF.h"
#include "SysLog.h"

using namespace std;


SDF::SDF(const string& source, const Mode mode)
: source(source), mode(mode), ia(0), oa(0) {

}

SDF::SDF(const SDF& sdf)
: SDF(sdf.source, sdf.mode) {}

SDF::~SDF() {
	sdf_close();
}

void SDF::open() {
	if (this->mode == NEW) {
		int mkdir_result = mkdir(this->source.c_str(), 0744);
		SASSERT(mkdir_result == 0, "mkdir %s failed.", this->source.c_str());
	}
	sdf_open();

	cout << "Opened sdf " << this->source << endl;


	/*
	if (this->mode == READ) {
		//this->ifs.seekg(0, ios::end);
		//this->dbSize = this->ifs.tellg();
		this->ifs.clear();
		this->ifs.seekg(0, ios::beg);
		cout << "Size of opend sdf is " << this->dbSize << endl;
	}
	*/
}

void SDF::close() {
	sdf_close();
}


void SDF::put(const string& key, const string& value) {
	SASSERT0(this->mode == NEW);

	this->keys.push_back(key);
	this->values.push_back(value);
}


const string SDF::getNextValue() {
	SASSERT0(this->mode == READ);
	SASSERT0(this->ifs.is_open());

	string key, value;
	(*this->ia) >> key >> value;
	if (this->ifs.tellg() == this->dbSize) {


		this->ifs.clear();
		this->ifs.seekg(0, ios::beg);


		// text_iarchive를 reset하는 더 좋은 방법이 있을 것 같다.
		if (this->ia) {
			delete this->ia;
			this->ia = 0;
		}
		unsigned int flags = boost::archive::no_header;
		this->ia = new boost::archive::text_iarchive(this->ifs, flags);

		string tKey, tValue;
		(*this->ia) >> tKey >> tValue;
		//cout << "end of SDF, reset to start point ...'" << tKey << "', '" << tValue << "'" << endl;
	}

	return value;
}


void SDF::commit() {
	SASSERT0(this->mode == NEW);
	SASSERT0(this->ofs.is_open());
	SASSERT0(this->keys.size() == this->values.size());

	for (int i = 0; i < this->keys.size(); i++) {
		(*this->oa) << this->keys[i] << this->values[i];
	}

	this->keys.clear();
	this->values.clear();
}




void SDF::sdf_open() {
	unsigned int flags = boost::archive::no_header;
	if (this->mode == NEW) {
		SASSERT0(!this->ofs.is_open());
		this->ofs.open(this->source + this->dataName, ios_base::out);
		this->oa = new boost::archive::text_oarchive(this->ofs, flags);
	} else if (this->mode == READ) {
		SASSERT0(!this->ifs.is_open());
		this->ifs.open(this->source + this->dataName, ios_base::in);






		this->ifs.seekg(0, ios::end);
		this->dbSize = this->ifs.tellg();
		this->ifs.clear();
		this->ifs.seekg(0, ios::beg);
		cout << "Size of opend sdf is " << this->dbSize << endl;



		this->ia = new boost::archive::text_iarchive(this->ifs, flags);
	} else {
		SASSERT0(false);
	}
}

void SDF::sdf_close() {
	if (this->ifs.is_open()) {
		this->ifs.close();
		if (this->ia) {
			delete this->ia;
			this->ia = 0;
		}
	}
	if (this->ofs.is_open()) {
		this->ofs.close();
		if (this->oa) {
			delete this->oa;
			this->oa = 0;
		}
	}
}















