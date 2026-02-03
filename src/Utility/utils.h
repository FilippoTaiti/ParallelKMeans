//
// Created by filippo on 15/01/26.
//

#ifndef KMEANSCLUSTERING_UTILS_H
#define KMEANSCLUSTERING_UTILS_H

#include "read_dataset.h"
#include <random>
#include <cfloat>

void print_vector(vector<float>& vector);

void kmeansplusplus(const Dataset_SoA &dataset, int k, mt19937_64 &generator, int n, float* __restrict squared_distances, vector<float>& centroids_x,
    vector<float>& centroids_y);

float mean(const vector<float>& vector);
float standard_dev(const vector<float>& vector, float mean);

bool isEqual(float a, float b);


#endif //KMEANSCLUSTERING_UTILS_H