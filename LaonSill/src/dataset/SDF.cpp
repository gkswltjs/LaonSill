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


void SDF::initHeader(SDFHeader& header) {
	SASSERT0(this->mode == NEW);
	SASSERT0(this->ofs.is_open());
	SASSERT0(header.numSets > 0);
	SASSERT0(this->header.numSets == 0);	// 아직까지 header가 initialize되지 않았어야 함

	this->header = header;
	(*this->oa) << header;

	this->bodyStartPos = this->ofs.tellp();
	this->currentPos.resize(header.numSets, 0);

	update_dataset_idx_map();
}

void SDF::updateHeader(SDFHeader& header) {
	SASSERT0(this->mode == NEW);
	SASSERT0(this->ofs.is_open());
	SASSERT0(header.numSets > 0);
	SASSERT0(this->header.numSets > 0);

	this->ofs.seekp(this->headerStartPos);
	this->header = header;
	(*this->oa) << header;

	SASSERT(this->bodyStartPos == this->ofs.tellp(), "bodyStartPos->%d, ofs.tellp->%d",
			this->bodyStartPos, this->ofs.tellp());
	this->currentPos = this->header.setStartPos;

	update_dataset_idx_map();
}

SDFHeader SDF::getHeader() {
	return this->header;
}

long SDF::getCurrentPos() {
	switch(this->mode) {
	case NEW:
		SASSERT0(this->ofs.is_open());
		return this->ofs.tellp();
	case READ:
		SASSERT0(this->ifs.is_open());
		return this->ifs.tellg();
	}
}

void SDF::setCurrentPos(long currentPos) {
	SASSERT0(this->mode == READ);
	SASSERT0(this->ifs.is_open());

	this->currentPos[this->curDataSetIdx] = currentPos;
	this->ifs.seekg(this->currentPos[this->curDataSetIdx], ios::beg);
}

int SDF::findDataSet(const string& dataSet) {
	auto itr = this->dataSetIdxMap.find(dataSet);
	if (itr == this->dataSetIdxMap.end()) {
		return -1;
	} else {
		return itr->second;
	}
}


void SDF::selectDataSet(const string& dataSet) {
	int dataSetIdx = findDataSet(dataSet);
	selectDataSet(dataSetIdx);
}

void SDF::selectDataSet(const int dataSetIdx) {
	SASSERT0(dataSetIdx >= 0 && dataSetIdx < this->dataSetIdxMap.size());

	//SASSERT0(this->mode == READ);
	//SASSERT0(this->ifs.is_open());

	this->curDataSetIdx = dataSetIdx;
	//this->ifs.seekg(this->currentPos[this->curDataSetIdx], ios::beg);

	setCurrentPos(this->currentPos[this->curDataSetIdx]);
}

const std::string& SDF::curDataSet() {
	SASSERT(this->curDataSetIdx >= 0, "Select dataset first ... ");
	return this->header.names[this->curDataSetIdx];
}


void SDF::put(const string& key, const string& value) {
	SASSERT0(this->mode == NEW);
	this->values.push_back(value);
}


const string SDF::getNextValue() {
	SASSERT0(this->mode == READ);
	SASSERT0(this->ifs.is_open());
	SASSERT0(this->curDataSetIdx >= 0);

	string value;
	(*this->ia) >> value;

	this->currentPos[this->curDataSetIdx] = this->ifs.tellg();
	long end = (this->curDataSetIdx >= this->header.numSets - 1) ?
			this->dbSize : this->header.setStartPos[this->curDataSetIdx + 1];

	if (this->currentPos[this->curDataSetIdx] >= end) {
		this->ifs.seekg(this->header.setStartPos[this->curDataSetIdx], ios::beg);
		this->currentPos[this->curDataSetIdx] = this->ifs.tellg();
	}

#if 0
	if (this->ifs.tellg() == this->dbSize) {
		/*
		this->ifs.clear();
		this->ifs.seekg(0, ios::beg);

		// text_iarchive를 reset하는 더 좋은 방법이 있을 것 같다.
		if (this->ia) {
			delete this->ia;
			this->ia = 0;
		}
#ifdef BOOST_ARCHIVE_NO_HEADER
		unsigned int flags = boost::archive::no_header;
#else
	unsigned int flags = 0;
#endif
		this->ia = new boost::archive::text_iarchive(this->ifs, flags);

		string tKey, tValue;
		(*this->ia) >> tKey >> tValue;
		//cout << "end of SDF, reset to start point ...'" << tKey << "', '" << tValue << "'" << endl;
		*/

		this->ifs.seekg(this->bodyStartPos, ios::beg);
	}
#endif

	return value;
}


void SDF::commit() {
	SASSERT0(this->mode == NEW);
	SASSERT0(this->ofs.is_open());

	for (int i = 0; i < this->values.size(); i++) {
		(*this->oa) << this->values[i];
	}
	this->values.clear();
}




void SDF::sdf_open() {
#ifdef BOOST_ARCHIVE_NO_HEADER
	unsigned int flags = boost::archive::no_header;
#else
	unsigned int flags = 0;
#endif
	if (this->mode == NEW) {
		SASSERT0(!this->ofs.is_open());
		this->ofs.open(this->source + this->DATA_NAME, ios_base::out);
		this->oa = new boost::archive::text_oarchive(this->ofs, flags);

		(*this->oa) << SDF_STRING;

		SDFHeader dummyHeader;
		(*this->oa) << dummyHeader;
		LabelItem dummyLabelItem;
		(*this->oa) << dummyLabelItem;

		this->headerStartPos = this->ofs.tellp();

	} else if (this->mode == READ) {
		SASSERT0(!this->ifs.is_open());
		this->ifs.open(this->source + this->DATA_NAME, ios_base::in);

		this->ifs.seekg(0, ios::end);
		this->dbSize = this->ifs.tellg();

		this->ifs.clear();
		this->ifs.seekg(0, ios::beg);
		cout << "Size of opend sdf is " << this->dbSize << endl;

		this->ia = new boost::archive::text_iarchive(this->ifs, flags);

		string sdf_string;
		(*this->ia) >> sdf_string;
		SASSERT0(sdf_string == SDF_STRING);
		(*this->ia) >> this->header;	// dummyHeader
		LabelItem dummyLabelItem;
		(*this->ia) >> dummyLabelItem;
		this->headerStartPos = this->ifs.tellg();
		(*this->ia) >> this->header;
		this->bodyStartPos = this->ifs.tellg();

		this->header.print();

		this->currentPos = this->header.setStartPos;
		update_dataset_idx_map();
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

void SDF::update_dataset_idx_map() {
	this->dataSetIdxMap.clear();
	for (int i = 0; i < this->header.names.size(); i++) {
		this->dataSetIdxMap[this->header.names[i]] = i;
	}
	this->curDataSetIdx = -1;
}












