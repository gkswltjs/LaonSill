/**
 * @file Timer.h
 * @date 2016/5/19
 * @author jhkim
 * @brief
 * @details
 */

#ifndef TIMER_H_
#define TIMER_H_

#include <iostream>

using namespace std;

/**
 * @brief 수행속도를 측정하기 위한 타이머 클래스
 */
class Timer {
public:
	/**
	 * @details Timer 기본 생성자
	 */
	Timer() { print = true; }
	/**
	 * @details Timer 소멸자
	 */
	virtual ~Timer() {}
	/**
	 * @details 시간 측정결과 출력여부를 설정한다.
	 *          현재는 사용하지 않는 method로 추정한다.
	 */
	void setPrint(bool print) { this->print = print; }
	/**
	 * @details 타이머 시간 측정을 시작한다.
	 */
	void start() {
		t1 = clock();
	}
	/**
	 * @details 타이머 시간 측정을 종료한다.
	 * @param print 측정 결과 출력 여부
	 * @return 측정된 시간
	 */
	long stop(bool print=true) {
		t2 = clock();
		long elapsed = t2 - t1;
		if(print) cout << "total " << elapsed << "ms elapsed ... " << endl;
		return elapsed;
	}

private:
	bool print;			///< 시간 측정결과 출력여부 (사용하지 않을 것으로 추정)
	clock_t t1;			///< 시간 측정 시작 시간
	clock_t t2;			///< 시간 측정 종료 시간
};

#endif /* TIMER_H_ */
