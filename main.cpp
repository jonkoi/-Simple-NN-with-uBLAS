/* *
 * Duy Khoi Tran
 * Simple Feed-forward neural network framework with uBLAS
 * Developed for the Master Thesis application requirement
 *
 * main.cpp
 */

#include "pch.h"
#include "Net.h"
#include <time.h>
#include <iostream>
#include "boost/numeric/ublas/io.hpp"

using namespace boost::numeric;

/**
 * Main function.
 * Toy example where only one neuron, being either more or less 
 * than 0.5, decides the output. 
 */
int main()
{
	std::cout << "Program Start!" << std::endl;
	const std::vector<int> sizes = { 2, 3, 3, 2 }; //! Changable
	Net net(sizes);
	ublas::vector<double> in(sizes.front()), out(sizes.back());
	
	int epoch = 0, c = 0;
	double correct = 0;
	while (epoch < 10) {
		in(0) = ((double)rand() / (RAND_MAX));
		in(1) = ((double)rand() / (RAND_MAX));

		out(0) = in(1) < 0.5 ? 1 : 0;
		out(1) = in(1) < 0.5 ? 0 : 1;
		int correct_neuron = in(1) < 0.5 ? 0 : 1;
		
		//Network process
		net.FeedForward(in);
		int prediction = net.GetMaxOutput();
		if (prediction == correct_neuron) correct++;
		net.BackPropagate(out);

		c++;
		if (c > 1000) {
			epoch++;
			double correct_ratio = correct / c;
			std::cout << "correct rate in epoch " << epoch << " is " << correct_ratio << std::endl;
			correct = 0;
			c = 0;
		}
	}

	std::cout << "Program End!" <<std::endl;
	system("pause");
	return 0;
}
