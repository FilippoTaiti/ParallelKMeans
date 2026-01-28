//
// Created by filippo on 15/01/26.
//

#ifndef KMEANSCLUSTERING_READ_DATASET_H
#define KMEANSCLUSTERING_READ_DATASET_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

using namespace std;

struct Dataset_SoA {
    vector<float> x; //100k
    vector<float> y; //100k
    vector<int> cluster_id; //100k

    explicit Dataset_SoA(const int n): x(n, 0), y(n, 0), cluster_id(n, -1) {}
};

Dataset_SoA read_csv(const string& name, int n, Dataset_SoA& dataset);








#endif //KMEANSCLUSTERING_READ_DATASET_H