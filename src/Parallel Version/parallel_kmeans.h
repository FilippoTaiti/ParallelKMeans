//
// Created by filippo on 15/01/26.
//

#ifndef KMEANSCLUSTERING_PARALLEL_KMEANS_H
#define KMEANSCLUSTERING_PARALLEL_KMEANS_H


#include "../Utility/read_dataset.h"

void parallel_kmeans(Dataset_SoA& dataset, int k, int number_of_iterations, int n, vector<float>&  centroids_x, vector<float>& centroids_y);


#endif //KMEANSCLUSTERING_PARALLEL_KMEANS_H

