#pragma once
#include <vector>
#include <iostream>
#include "layer.hpp"


class Sequence {
    public:
        std::vector<Dense> dense_layers;
        std::vector<Flatten> flatten_layers;
        Sequence();
        void compile(std::string optimizer, std::string loss, std::vector<std::string> metrics);
        void addLayer(Dense layer);
        void addLayer(Flatten layer);
};
