#pragma once
#include <vector>
#include <iostream>

class Neuron {
    public:
        std::vector<double> inputs;
        std::vector<double> weights;
        double bias;
        Neuron();
        Neuron(std::vector<double> inputs, std::vector<double> weights, double bias);
        double output();
};
