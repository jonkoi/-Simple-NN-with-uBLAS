#pragma once
/* *
 * Duy Khoi Tran
 * Simple Feed-forward neural network framework with uBLAS
 * Developed for the Master Thesis application requirement
 *
 * Net.h
 */
#include <vector>
#include <random>
#include <cmath>
#include <numeric>
#include "boost\numeric\ublas\matrix.hpp"
#include "boost\numeric\ublas\vector.hpp"
#include "boost\numeric\ublas\io.hpp"
#include "pch.h"

using namespace boost::numeric;

class Net
{
public:
	Net(const std::vector<int> &sizes);

	void FeedForward(ublas::vector<double> &x);
	void BackPropagate(ublas::vector<double> &y);
	int GetMaxOutput();

private:
	int nLayers;
	double learning_rate;
	std::vector<ublas::matrix<double>> weights;
	std::vector<ublas::vector<double>> neuron_vals;
	std::vector<ublas::vector<double>> errors; //! Can be made omitted

	ublas::vector<double> ActivationDerivative(ublas::vector<double> &o);
};

