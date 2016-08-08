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
#include "MonitorSet.h"
#include <gnuplot-iostream.h>
#include <cmath>
#include <boost/tuple/tuple.hpp>

using namespace std;


/**
 * @brief 네트워크 상태 모니터링 클래스
 * @details 네트워크의 이벤트마다 이벤트 파라미터를 그래프로 ploat하는 역할을 한다.
 */
class NetworkMonitor : public NetworkListener {
public:
	NetworkMonitor(int maxEpoch)
		: maxEpoch(maxEpoch) {

		//gnuplot_accuracy << "set xrange [0:30]\nset yrange [80:100]\n";
		//gnuplot_cost << "set xrange [0:30]\nset yrange [0:0.5]\n";
		//gnuplot << "plot '-' with lines title 'accuracy'" << "\n";
		//gnuplot.send1d(accuracySet);
	}
	virtual ~NetworkMonitor() {
		//gnuplot_accuracy.close();
		//gnuplot_cost.close();
	}

	//void epochComplete(double validationCost, double validationAccuracy, double trainCost, double trainAccuracy) {
	void epochComplete(float cost, float accuracy) {
		if(accuracy > accuracyMax) accuracyMax = accuracy;
		if(accuracy < accuracyMin) accuracyMin = accuracy;
		if(cost > costMax) costMax = cost;
		if(cost < costMin) costMin = cost;

		accuracySet.push_back(boost::make_tuple(accuracySet.size()+1, accuracy*100));
		costSet.push_back(boost::make_tuple(costSet.size()+1, cost));

		//gnuplot_accuracy << "set xrange [0:30]\nset yrange [80:100]\n";
		//gnuplot_cost << "set xrange [0:30]\nset yrange [0:0.5]\n";

		if(accuracyMax != 0 || accuracyMin != 0) {
			gnuplot_accuracy << "set xrange [0:" << std::min(maxEpoch+1, (int)(accuracySet.size()+10))
				<< "]\nset yrange [" << std::max(0.0, 80.0*accuracyMin) << ":" << std::min(100.0, 120.0*accuracyMax) << "]\n";
			gnuplot_accuracy << "plot '-' with linespoints pt 1 title 'accuracy'" << "\n";
			gnuplot_accuracy.send1d(accuracySet);
		}

		if(costMax != 0 > costMin != 0) {
			gnuplot_cost << "set xrange [0:" << std::min(maxEpoch+1, (int)(costSet.size()+10))
				<< "]\nset yrange [" << std::max(0.0, 0.95*costMin) << ":" << std::min(2.0, 1.05*costMax) << "]\n";
			gnuplot_cost << "plot '-' with linespoints pt 1 title 'cost'" << "\n";
			gnuplot_cost.send1d(costSet);
		}

		/*
		// Create a script which can be manually fed into gnuplot later:
		//    Gnuplot gp(">script.gp");
		// Create script and also feed to gnuplot:
		//    Gnuplot gp("tee plot.gp | gnuplot -persist");
		// Or choose any of those options at runtime by setting the GNUPLOT_IOSTREAM_CMD
		// environment variable.

		// Gnuplot vectors (i.e. arrows) require four columns: (x,y,dx,dy)
		std::vector<boost::tuple<double, double, double, double> > pts_A;

		// You can also use a separate container for each column, like so:
		std::vector<double> pts_B_x;
		std::vector<double> pts_B_y;
		std::vector<double> pts_B_dx;
		std::vector<double> pts_B_dy;

		// You could also use:
		//   std::vector<std::vector<double> >
		//   boost::tuple of four std::vector's
		//   std::vector of std::tuple (if you have C++11)
		//   arma::mat (with the Armadillo library)
		//   blitz::Array<blitz::TinyVector<double, 4>, 1> (with the Blitz++ library)
		// ... or anything of that sort

		for(double alpha=0; alpha<1; alpha+=1.0/24.0) {
			double theta = alpha*2.0*3.14159;
			pts_A.push_back(boost::make_tuple(
				 cos(theta),
				 sin(theta),
				-cos(theta)*0.1,
				-sin(theta)*0.1
			));

			pts_B_x .push_back( cos(theta)*0.8);
			pts_B_y .push_back( sin(theta)*0.8);
			pts_B_dx.push_back( sin(theta)*0.1);
			pts_B_dy.push_back(-cos(theta)*0.1);
		}
		*/

		// Don't forget to put "\n" at the end of each line!
		//gnuplot << "set xrange [0:10]\nset yrange [0:100]\n";
		// '-' means read from stdin.  The send1d() function sends data to gnuplot's stdin.
		//gnuplot << "plot '-' with vectors title 'pts_A', '-' with vectors title 'pts_B'\n";
		//gnuplot.send1d(pts_A);
		//gnuplot.send1d(boost::make_tuple(pts_B_x, pts_B_y, pts_B_dx, pts_B_dy));


	}

protected:
	vector<boost::tuple<int, float>> accuracySet;		///< epoch마다 계산된 #epoch, 정확도 튜플 목록 벡터
	vector<boost::tuple<int, float>> costSet;			///< epcoh마다 계산된 #epcoh, cost 튜플 목록 벡터
	float accuracyMax=0.0f;								///< 학습의 정확도 최대값
	float accuracyMin=1.0f;								///< 학습의 정확도 최소값
	float costMax=0.0f;									///< 학습의 cost 최대값
	float costMin=100.0f;								///< 학습의 cost 최소값

	Gnuplot gnuplot_accuracy;							///< gnu plot 정확도 그래프 객체
	Gnuplot gnuplot_cost;								///< gnu plot cost 그래프 객체

	const int maxEpoch;									///< 최대 epoch값

};

#endif /* MONITOR_NETWORKMONITOR_H_ */
