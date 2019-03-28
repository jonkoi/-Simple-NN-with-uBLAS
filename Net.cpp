/* *
 * Duy Khoi Tran
 * Simple Feed-forward neural network framework with uBLAS
 * Developed for the Master Thesis application requirement
 *
 * Net.cpp
 */
#include "Net.h"

using namespace boost::numeric;

/**
 * Constructor of Net class.
 * Initialize and store temporary zero uBLAS containers.
 * Weights initialized with normal normal distribution.
 */
Net::Net(const std::vector<int> &sizes):
	learning_rate(1.0)
{
	//Placeholders
	nLayers = static_cast<int>(sizes.size());
	for (int i = 0; i < nLayers; i++) {
		neuron_vals.push_back(ublas::zero_vector<double>(sizes[i]));
	}
	for (int i = 1; i < nLayers; i++) {
		weights.push_back(ublas::zero_matrix<double>(sizes[i], sizes[i - 1]));
		errors.push_back(ublas::zero_vector<double>(sizes[i]));
	}

	// Weight Initialization
	// Reference: https://github.com/GarethRichards/Machine-Learning-CPP
	std::random_device rd;
	std::mt19937 gen(rd());
	for (auto &w : weights) {
		std::normal_distribution<> d(0, 1);
		for (auto &e : w.data()) {
			e = d(gen);
		}
	}
}

/**
 * Feed-forward process.
 * Sigmoid-activation with Cross Entropy Loss.
 */
void Net::FeedForward(ublas::vector<double> &x) {
	neuron_vals[0] = x;
	for (int i = 0; i < nLayers - 1; i++) {
		neuron_vals[i+1] = prod(weights[i], neuron_vals[i]);
		for (auto &output : neuron_vals[i + 1]) {
			output = 1.0 / (1.0 + exp(-output));
		}
	}
}

/**
 * Backprop process.
 */
void Net::BackPropagate(ublas::vector<double> &y) {
	// Errors cumulation
	for (int i = nLayers-1; i > 0; i--) {
		ublas::vector<double> derivative = ActivationDerivative(neuron_vals[i]);
		if (i == nLayers-1) {
			errors[i - 1] = element_prod(ActivationDerivative(neuron_vals[i]), 
				(y - neuron_vals[i]));
		}
		else
		{
			errors[i - 1] = element_prod(ActivationDerivative(neuron_vals[i]), 
				prod(trans(weights[i]), errors[i]));
		}
	}
	
	//Weight update
	for (int i = nLayers - 2; i >= 0; i--) {
		ublas::matrix<double> dw = learning_rate * outer_prod(errors[i], neuron_vals[i]);
		weights[i] += dw;
	}
}

/**
 * Derative of sigmoid
 */
ublas::vector<double> Net::ActivationDerivative(ublas::vector<double> &o) {
	ublas::vector<double> derivative(o.size());
	for (int i = 0; i < o.size(); i++) {
		derivative(i) = o(i) * (1.0 - o(i));
	}
	return derivative;
}

/**
 * Get index of largest output
 */
int Net::GetMaxOutput() {
	double cur_max = -100.0;
	int cur_max_id = 0;
	for (int i = 0; i < neuron_vals.back().size(); i++) {
		if (neuron_vals.back()(i) > cur_max) {
			cur_max = neuron_vals.back()(i);
			cur_max_id = i;
		}
	}
	return cur_max_id;
}