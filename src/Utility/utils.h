//
// Created by filippo on 15/01/26.
//

#ifndef KMEANSCLUSTERING_UTILS_H
#define KMEANSCLUSTERING_UTILS_H

#include "read_dataset.h"
#include <random>
#include <cfloat>

void print_vector(vector<float>& vector);

void kmeansplusplus(const Dataset_SoA &dataset, int k, mt19937_64 &generator, int n, float* __restrict squared_distances, float* __restrict__ centroids_x,
    float* __restrict__ centroids_y);

double mean(const vector<double>& vector);
double standard_dev(const vector<double>& vector, double mean);



#endif //KMEANSCLUSTERING_UTILS_H