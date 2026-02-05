//
// Created by filippo on 15/01/26.
//

#ifndef KMEANSCLUSTERING_SEQUENTIAL_KMEANS_H
#define KMEANSCLUSTERING_SEQUENTIAL_KMEANS_H


#include "../Utility/read_dataset.h"

void sequential_kmeans(Dataset_SoA &dataset, int k, int number_of_iterations, int n, float* __restrict__ centroids_x, float* __restrict__ centroids_y);



#endif //KMEANSCLUSTERING_SEQUENTIAL_KMEANS_H