/**
 * @file NetworkMonitor.h
 * @date 2016/4/26
 * @author jhkim
 * @brief
 * @details
 */

#ifndef MONITOR_NETWORKMONITOR_H_
#define MONITOR_NETWORKMONITOR_H_

#include <iostream>

#include "NetworkListener.h"
#include <gnuplot-iostream.h>
#include <cmath>
#include <cfloat>
#include <boost/tuple/tuple.hpp>
#include "../debug/StatLogger.h"
#include "../debug/StatGraphPlotter.h"
#include "../debug/StatFileWriter.h"

using namespace std;


/**
 * @brief 네트워크 상태 모니터링 클래스
 * @details 네트워크의 이벤트마다 이벤트 파라미터를 그래프로 plot하는 역할을 한다.
 */
class NetworkMonitor : public NetworkListener {
public:
	static const int PLOT_ONLY = 0;
	static const int WRITE_ONLY = 1;
	static const int PLOT_AND_WRITE = 2;

	NetworkMonitor(int mode = PLOT_ONLY)
		: mode(mode) {

		if(mode != PLOT_ONLY &&
				mode != WRITE_ONLY &&
				mode != PLOT_AND_WRITE) {
			cout << "invalid monitor mode ... " << endl;
			exit(1);
		}

		loggerSize = 1;
		if(mode == PLOT_AND_WRITE) loggerSize = 2;

		if(mode == PLOT_ONLY || mode == PLOT_AND_WRITE) {
			costLogger.push_back(new StatGraphPlotter());
			accuracyLogger.push_back(new StatGraphPlotter());
			gradSumsqLogger.push_back(new StatGraphPlotter(250, 10));
			dataSumsqLogger.push_back(new StatGraphPlotter(250, 10));
		}
		if(mode == WRITE_ONLY || mode == PLOT_AND_WRITE) {
			const string postfix = now();

			costLogger.push_back(new StatFileWriter("./cost_"+postfix+".csv"));
			accuracyLogger.push_back(new StatFileWriter("./accuracy_"+postfix+".csv"));
			gradSumsqLogger.push_back(new StatFileWriter("./gradSumsq_"+postfix+".csv"));
			dataSumsqLogger.push_back(new StatFileWriter("./dataSumsq_"+postfix+".csv"));
		}
	}
	virtual ~NetworkMonitor() {
		for(uint32_t i = 0; i < loggerSize; i++) {
			delete costLogger[i];
			delete accuracyLogger[i];
			delete gradSumsqLogger[i];
			delete dataSumsqLogger[i];
		}
		costLogger.clear();
		accuracyLogger.clear();
		gradSumsqLogger.clear();
		dataSumsqLogger.clear();
	}

	void onCostComputed(const uint32_t index, const string name, const double cost) {
		for(uint32_t i = 0; i < loggerSize; i++) {
			costLogger[i]->addStat(index, name, cost);
		}
	}
	void onAccuracyComputed(const uint32_t index, const string name, const double accuracy) {
		for(uint32_t i = 0; i < loggerSize; i++) {
			accuracyLogger[i]->addStat(index, name, accuracy);
		}
	}
	void onGradSumsqComputed(const uint32_t index, const string name, const double sumsq) {
		for(uint32_t i = 0; i < loggerSize; i++) {
			gradSumsqLogger[i]->addStat(index, name, sumsq);
		}
	}
	void onDataSumsqComputed(const uint32_t index, const string name, const double sumsq) {
		for(uint32_t i = 0; i < loggerSize; i++) {
			dataSumsqLogger[i]->addStat(index, name, sumsq);
		}
	}


private:
	const string now() {
		// current date/time based on current system
		time_t now = time(0);
		//cout << "Number of sec since January 1,1970:" << now << endl;
		tm *ltm = localtime(&now);

		// print various components of tm structure.
		//cout << "Year" << 1900 + ltm→tm_year<<endl;
		//cout << "Month: "<< 1 + ltm->tm_mon<< endl;
		//cout << "Day: "<<  ltm->tm_mday << endl;
		//cout << "Time: "<< 1 + ltm->tm_hour << ":";
		//cout << 1 + ltm->tm_min << ":";
		//cout << 1 + ltm->tm_sec << endl;

		return to_string(1900+ltm->tm_year)+
				to_string(1+ltm->tm_mon)+
				to_string(ltm->tm_mday)+
				to_string(1+ltm->tm_hour)+
				to_string(1+ltm->tm_min)+
				to_string(1+ltm->tm_sec);

	}


protected:
	uint32_t mode;
	uint32_t loggerSize;

	vector<StatLogger*> accuracyLogger;
	vector<StatLogger*> costLogger;
	vector<StatLogger*> gradSumsqLogger;
	vector<StatLogger*> dataSumsqLogger;

	/*
	GraphPlotter accuracyPlotter;
	GraphPlotter costPlotter;
	GraphPlotter gradSumsqPlotter;
	GraphPlotter dataSumsqPlotter;

	GraphWriter accuracyWriter;
	GraphWriter costWriter;
	GraphWriter gradSumsqWriter;
	GraphWriter dataSumsqWriter;
	*/


};

#endif /* MONITOR_NETWORKMONITOR_H_ */



























