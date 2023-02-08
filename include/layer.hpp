#pragma once
#include <vector>
#include <sstream>
#include <iomanip>
#include <random> 
#include <type_traits>
#include "activation.hpp"


class Dense {
    public:
        std::tuple<int, int> shape;
        std::vector<std::vector<double>> weights;
        std::vector<double> biases;
        Activation activation;
        Dense();
        Dense(int nInputs, int nNeurons, std::string activation);
        std::vector<std::vector<double>> forward(std::vector<std::vector<double>> inputs);
        void print();
    private:
        std::vector<std::vector<double>> _initializeWeights(std::tuple<int, int> shape);
};

class Flatten {
    public:
        Flatten();
        void print();
};

