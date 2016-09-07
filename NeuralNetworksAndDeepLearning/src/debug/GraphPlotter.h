/*
 * GraphPlotter.h
 *
 *  Created on: 2016. 9. 2.
 *      Author: jhkim
 */

#ifndef GRAPHPLOTTER_H_
#define GRAPHPLOTTER_H_

#include <iostream>

#include <gnuplot-iostream.h>
#include <boost/tuple/tuple.hpp>
#include <cmath>
#include <cfloat>


using namespace std;


class GraphPlotter {
public:
	GraphPlotter(uint32_t windowSize=500, uint32_t refreshFreq=1)
		: windowSize(windowSize), refreshFreq(refreshFreq) {
		statMax = DBL_MIN;
		statMin = DBL_MAX;

		refreshCount = 0;
		statCount = 1;
	}
	/*
	GraphPlotter(const vector<string>& statNameList, uint32_t windowSize=500, uint32_t refreshFreq=1)
		: GraphPlotter(windowSize, refreshFreq) {

		if(statCount != statNameList.size()) {
			cout << "statCount != statNameList.size() ... " << endl;
			exit(1);
		}
		this->statNameList = statNameList;
		statList.resize(statCount);
	}
	*/
	virtual ~GraphPlotter() {}

	void addStat(const uint32_t statIndex, const string statName, const double stat) {
		//cout << "statIndex: " << statIndex << ", statName: " << statName << ", stat: " << stat << endl;


		if(stat > statMax) statMax = stat;
		if(stat < statMin) statMin = stat;

		// 새로운 아이템 index 등장, 추가
		if(statIndex == statList.size()) {
			statList.push_back(vector<boost::tuple<int, double>>());
			//statList[statIndex].push_back(boost::make_tuple(statList[statIndex].size(), stat));
			statList[statIndex].push_back(boost::make_tuple(0, stat));
			statNameList.push_back(statName);
			return;
		}
		// 연속되지 않은 새로운 아이템 index 등장, error로 처리
		else if(statIndex > statList.size()) {
			cout << "invalid stat index ... " << endl;
			exit(1);
		}


		// 새로운 아이템 인덱스가 아닌 경우/////////////////////////////////////////////////////////////////////////


		if(statList[statIndex].size() >= windowSize) {
			statList[statIndex].erase(statList[statIndex].begin());
		}
		//statList[statIndex].push_back(boost::make_tuple(statList[statIndex].size(), stat));
		statList[statIndex].push_back(boost::make_tuple(statCount, stat));

		if(statIndex < statList.size()-1) return;
		statCount++;

		if(refreshFreq > 1 && (refreshCount++ % refreshFreq) != 0) return;

		double localStatMax = DBL_MIN;
		double localStatMin = DBL_MAX;
		for(uint32_t i = 0; i < statList.size(); i++) {
			for(uint32_t j = std::max(0, (int)(statList[i].size()-windowSize)); j < statList[i].size(); j++) {
				if(statList[i][j].get<1>() > localStatMax) {
					localStatMax = statList[i][j].get<1>();
				}
				if(statList[i][j].get<1>() < localStatMin) {
					localStatMin = statList[i][j].get<1>();
				}
			}
		}

		//int statItemSize = statList[statIndex].size();
		int statItemSize = statCount;
		plot << "set xrange [" << std::max(0, (int)(statItemSize-windowSize)) << ":" << statItemSize+5 << "]\nset yrange [" << std::max(0.0, 0.95*localStatMin) << ":" << 1.05*localStatMax << "]\n";
		plot << "plot ";
		for(uint32_t i = 0; i < statList.size(); i++) {
			plot << plot.file1d(statList[i]) << " with linespoints pt 1 title '"<< statNameList[i] << "'";
			if(i < statList.size()-1) plot << ",";
			else plot << endl;
		}

		//cout << "drawed graph ... " << endl;
	}


private:
	vector<vector<boost::tuple<int, double>>> statList;
	vector<string> statNameList;
	Gnuplot plot;

	double statMax;
	double statMin;


	const uint32_t windowSize;
	const uint32_t refreshFreq;
	uint32_t statCount;
	uint32_t refreshCount;

};

#endif /* GRAPHPLOTTER_H_ */
