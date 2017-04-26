/*
 * LMDBDataSet.cpp
 *
 *  Created on: Mar 22, 2017
 *      Author: jkim
 */

#include "LMDBDataSet.h"

using namespace std;

template <typename Dtype>
LMDBDataSet<Dtype>::LMDBDataSet(const string& source)
: _source(source) {

	MDB_env* env_handle{nullptr};
	lmdb::env_create(&env_handle);
	this->_env = new lmdb::env(env_handle);
	this->_env->open(this->_source.c_str());

	MDB_txn* txn_handle{nullptr};
	lmdb::txn_begin(*(this->_env), nullptr, MDB_RDONLY, &txn_handle);
	this->_rtxn = new lmdb::txn(txn_handle);

	MDB_dbi dbi_handle{};
	lmdb::dbi_open(*(this->_rtxn), nullptr, 0, &dbi_handle);
	this->_dbi = new lmdb::dbi(dbi_handle);

	MDB_cursor* cursor_handle{};
	lmdb::cursor_open(*(this->_rtxn), *(this->_dbi), &cursor_handle);
	this->_cursor = new lmdb::cursor(cursor_handle);


	// XXX: ilsvrc12 데이터만 가정
	// 나중에 데이터에 따라 flexible하게 생성하도록 수정해야 함.
	this->rows = 224;
	this->cols = 224;
	this->channels = 3;

	//this->rows = 8;
	//this->cols = 12;
	//this->channels = 3;

	this->dataSize = this->rows * this->cols * this->channels;
	this->numTrainData = this->_dbi->size(*(this->_rtxn));
	cout << "numTrainData: " << this->numTrainData << endl;

	this->_trainData = new Dtype[this->dataSize];
	this->_trainLabel = new Dtype[1];
	this->_isRetrieved = true;

}

template <typename Dtype>
LMDBDataSet<Dtype>::~LMDBDataSet() {
	this->_cursor->close();
	this->_rtxn->abort();
	this->_env->close();

	if (this->_cursor)
		delete this->_cursor;
	if (this->_dbi)
		delete this->_dbi;
	if (this->_rtxn)
		delete this->_rtxn;
	if (this->_env)
		delete this->_env;

	if (this->_trainData)
		delete this->_trainData;
	if (this->_trainLabel)
		delete this->_trainLabel;
}

template <typename Dtype>
void LMDBDataSet<Dtype>::load() {







}

template <typename Dtype>
const Dtype* LMDBDataSet<Dtype>::getTrainDataAt(int index) {
	// XXX: 무조건 data 조회, label 조회 순을 전제하고 있음.
	assert(this->_isRetrieved == true);
	this->_isRetrieved = false;

	string key, value;
	if (!this->_cursor->get(key, value, MDB_NEXT)) {
		// end of data set
		cout << "end of cursor ... " << endl;
		this->_cursor->renew(*(this->_rtxn));

		if (!this->_cursor->get(key, value, MDB_NEXT)) {
			cout << "cannot retrieve data from LMDB ... " << endl;
			exit(1);
		}
	}

	//cout << "key: " << key << endl;

	// train data 조회
	unsigned char* ptr = (unsigned char*)value.c_str();
	for (int i = 0; i < this->dataSize; i++) {
	   	this->_trainData[i] = Dtype(ptr[12+i]);
	}

	// train label 조회
	// label정보가 1byte로 구성된 경우
	uint16_t category = 0;
	const int valueSize = value.size();
	if (valueSize == 12 + this->dataSize + 4) {
		category = ptr[valueSize-3];
	}
	// label정보가 2byte로 구성된 경우
	else if (valueSize + this->dataSize + 5) {
		category = (0x7f & ptr[valueSize-4]);
		category = (category | (ptr[valueSize-3] << 7));
	}
	else {
		cout << "invalid value length ... " << endl;
		exit(1);
	}

	assert(category < 1000);
	this->_trainLabel[0] = Dtype(category);

	return this->_trainData;
}

template <typename Dtype>
const Dtype* LMDBDataSet<Dtype>::getTrainLabelAt(int index) {
	assert(this->_isRetrieved == false);

	this->_isRetrieved = true;
	return this->_trainLabel;
}

template <typename Dtype>
const Dtype* LMDBDataSet<Dtype>::getValidationDataAt(int index) {

}


template <typename Dtype>
const Dtype* LMDBDataSet<Dtype>::getValidationLabelAt(int index) {

}

template <typename Dtype>
const Dtype* LMDBDataSet<Dtype>::getTestDataAt(int index) {

}

template <typename Dtype>
const Dtype* LMDBDataSet<Dtype>::getTestLabelAt(int index) {

}

template <typename Dtype>
void LMDBDataSet<Dtype>::shuffleTrainDataSet() {
	this->_cursor->renew(*(this->_rtxn));
}

template <typename Dtype>
void LMDBDataSet<Dtype>::shuffleValidationDataSet() {

}

template <typename Dtype>
void LMDBDataSet<Dtype>::shuffleTestDataSet() {

}


template class LMDBDataSet<float>;
