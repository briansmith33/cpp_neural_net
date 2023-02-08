#include "../include/sequence.hpp"


Sequence::Sequence() {}

void Sequence::compile(std::string optimizer, std::string loss, std::vector<std::string> metrics) {
    std::cout << dense_layers.size() << std::endl;
    for (int i = 0; i < dense_layers.size(); i++) {
        dense_layers[i].print();
    }
    
    std::cout << flatten_layers.size() << std::endl;
    for (int i = 0; i < flatten_layers.size(); i++) {
        flatten_layers[i].print();
    }
}

void Sequence::addLayer(Dense layer) {
    dense_layers.push_back(layer);
}

void Sequence::addLayer(Flatten layer) {
    flatten_layers.push_back(layer);
}

