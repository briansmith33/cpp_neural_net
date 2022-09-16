#include "../include/neuron.hpp"

Neuron::Neuron() {}

Neuron::Neuron(std::vector<double> i, std::vector<double> w, double b) {
    inputs = i;
    weights = w;
    bias = b;
}

double Neuron::output() {
    double o = 0;
    for (int i = 0; i < inputs.size(); i++) {
        o += inputs[i] * weights[i];
    }
    return o + bias;
}
