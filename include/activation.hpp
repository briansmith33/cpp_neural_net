#pragma once
#include <sstream>
#include <unordered_map>
#include <iostream>
#include <cmath>
#include <numeric>


class Activation {
    public:
        int function;
        Activation();
        Activation(std::string function);
        double calculate(double input);
        std::vector<std::vector<double>> softmax(std::vector<std::vector<double>> input);
    private:
        double _relu(double input);
        double _step(double input);
        double _sigmoid(double input);
        double _softmax(double input);
        
};
