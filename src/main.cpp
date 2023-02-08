#include <iostream>
#include <vector>
#include "layer.cpp"
#include "sequence.cpp"
#include <iomanip>
#include <random> 


std::vector<std::vector<double>> getData() {
    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<double> distr(-1, 1);
    std::vector<std::vector<double>> data (100, std::vector<double> (2, 0));
    for (auto &item : data) {
        for (auto &item1 : item) {
            item1 = distr(eng);
        }
    }
    return data;
}


int main()
{
    Sequence model = Sequence();
    model.addLayer(Dense(3, 5, "relu"));
    model.addLayer(Dense(5, 5, "relu"));
    model.addLayer(Flatten());
    model.compile("podspd", "npdonspd", {"nsdiopnso"});

    return EXIT_SUCCESS;
}

/*
std::vector<std::vector<double>> data = getData();
Dense layer1 = Dense(3, 5, "relu");
for (auto &item : layer1.forward(data)) {
    for (auto &elem : item)
        std::cout << std::setw(2) << elem << "; ";
        std::cout << std::endl;
}
std::cout << std::endl;
*/
