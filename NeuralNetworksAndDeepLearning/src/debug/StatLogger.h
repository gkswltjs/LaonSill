/*
 * StatLogger.h
 *
 *  Created on: 2016. 9. 10.
 *      Author: jhkim
 */

#ifndef STATLOGGER_H_
#define STATLOGGER_H_

class StatLogger {
public:
	StatLogger() {}
	virtual ~StatLogger() {}

	virtual void addStat(const uint32_t statIndex, const string statName, const double stat) = 0;
};

#endif /* STATLOGGER_H_ */
