#include "../include/activation.hpp"

static std::unordered_map<std::string, int> const table = { 
    {"relu", 1}, 
    {"step", 2}, 
    {"sigmoid", 3}, 
    {"softmax", 4} 
};

const double E = 2.718281828459045;

Activation::Activation() {}

Activation::Activation(std::string f) {
    auto it = table.find(f);
    if (it != table.end()) {
        function = it->second;
    } else { 
        std::cout << "Error\n";
     }
}

double Activation::calculate(double input) {
    switch (function) {
        case 1:
            return _relu(input);
        case 2:
            return _step(input);
        case 3:
            return _sigmoid(input);
        case 4:
            return _softmax(input);
        default:
            std::cout << "Error\n";
            return 0;
    }
}

std::vector<std::vector<double>> Activation::softmax(std::vector<std::vector<double>> input) {
    std::vector<std::vector<double>> output;
    for (int i = 0; i < input.size(); i++) {
        std::vector<double> row;
        output.push_back(row);
        for (int j = 0; j < input[i].size(); j++) {
            output[i].push_back(input[i][j] / accumulate(input[i].begin(), input[i].end(), 0));
        }
    }
    return output;
}

double Activation::_relu(double input) {
    if (input <= 0) {
        return 0;
    } else {
        return input;
    }
}

double Activation::_step(double input) {
    if (input <= 0) {
        return 0;
    } else {
        return 1;
    }
}

double Activation::_sigmoid(double input) {
    return 1 / (1 + std::pow(E, -input));
}

double Activation::_softmax(double input) {
    return std::pow(E, input);
}
