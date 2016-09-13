/*
 * StatFileWriter.h
 *
 *  Created on: 2016. 9. 10.
 *      Author: jhkim
 */

#ifndef STATFILEWRITER_H_
#define STATFILEWRITER_H_

#include "StatLogger.h"
#include <string>
#include <iostream>
#include <fstream>


using namespace std;

class StatFileWriter : public StatLogger {
public:
	StatFileWriter(const string& filename)
		: outstream(filename.c_str(), ios::out | ios::binary) {
		initialized = false;
	}
	virtual ~StatFileWriter() {}

	void addStat(const uint32_t statIndex, const string statName, const double stat) {

		if(statIndex == statNameList.size()) {
			statNameList.push_back(statName);
			statList.push_back(stat);
			return;
		} else if(statIndex > statNameList.size()) {
			cout << "invalid stat index ... " << endl;
			exit(1);
		}


		if(!initialized) {
			writeHeader();
			writeStat();
			initialized = true;
		}
		statList.push_back(stat);
		if(statIndex < statNameList.size()-1) return;

		writeStat();
	}


private:
	void writeHeader() {
		for(uint32_t i = 0; i < statNameList.size(); i++) {
			if(i < statNameList.size()-1) {
				outstream << statNameList[i] << ",";
			} else {
				outstream << statNameList[i] << endl;
			}
		}
	}

	void writeStat() {
		for(uint32_t i = 0; i < statList.size(); i++) {
			if(i < statList.size()-1) {
				outstream << statList[i] << ",";
			} else {
				outstream << statList[i] << endl;
			}
		}
		statList.clear();
	}

	vector<string> statNameList;
	vector<double> statList;
	ofstream outstream;

	uint32_t statSize;
	bool initialized;

};

#endif /* STATFILEWRITER_H_ */
