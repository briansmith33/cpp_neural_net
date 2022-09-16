#include "../include/layer.hpp"
#include "activation.cpp"
#include "neuron.cpp"

Dense::Dense() {}

Dense::Dense(int nInputs, int nNeurons, std::string a) {
    shape = {nInputs, nNeurons};
    weights = _initializeWeights(shape);
    std::vector<double> zeros(nNeurons, 0);
    biases = zeros;
    activation = Activation(a);
}

std::vector<std::vector<double>> Dense::forward(std::vector<std::vector<double>> inputs) {
    std::vector<std::vector<double>> output;
    for (int i = 0; i < inputs.size(); i++) {
        std::vector<double> row;
        output.push_back(row);
        for (int j = 0; j < weights.size(); j++) {
            output[i].push_back(activation.calculate(Neuron(inputs[i], weights[j], biases[j]).output()));
        }
    }
    if (activation.function == 4) {
        output = activation.softmax(output);
    }
    return output;
}

std::vector<std::vector<double>> Dense::_initializeWeights(std::tuple<int, int> shape) {
    int nInputs;
    int nNeurons;
    std::tie (nInputs, nNeurons) = shape;

    constexpr int MIN = -1;
    constexpr int MAX = 1;
    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<double> distr(MIN, MAX);
    std::vector<std::vector<double>> tensor (nInputs, std::vector<double> (nNeurons, 0));
    for (auto &item : tensor) {
        for (auto &item1 : item) {
            item1 = distr(eng);
        }
    }
    return tensor;
}

