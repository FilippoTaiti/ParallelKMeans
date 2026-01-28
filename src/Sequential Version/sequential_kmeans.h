//
// Created by filippo on 15/01/26.
//

#ifndef KMEANSCLUSTERING_SEQUENTIAL_KMEANS_H
#define KMEANSCLUSTERING_SEQUENTIAL_KMEANS_H


#include "../Utility/read_dataset.h"

void sequential_kmeans(Dataset_SoA &dataset, int k, int number_of_iterations, int n, vector<float>& centroids_x, vector<float>& centroids_y,
    int* __restrict number_of_elements_in_a_cluster, float* __restrict sum_x, float* __restrict sum_y);



#endif //KMEANSCLUSTERING_SEQUENTIAL_KMEANS_H