/*
 * Timer.h
 *
 *  Created on: 2016. 5. 19.
 *      Author: jhkim
 */

#ifndef TIMER_H_
#define TIMER_H_

#include <iostream>

using namespace std;


class Timer {
public:
	Timer() { print = true; }
	virtual ~Timer() {}

	void setPrint(bool print) { this->print = print; }

	void start() {
		t1 = clock();
	}

	void stop() {
		t2 = clock();
		cout << "total " << t2-t1 << "ms elapsed ... " << endl;
	}

private:
	bool print;
	clock_t t1;
	clock_t t2;
};

#endif /* TIMER_H_ */
