//
// Created by filippo on 15/01/26.
//

#include "sequential_kmeans.h"

#include <algorithm>
#include <set>
#include <cfloat>
#include <chrono>
#include <cstring>
#include <random>

#include "../Utility/utils.h"

void sequential_kmeans(Dataset_SoA &dataset, const int k, const int number_of_iterations, const int n, vector<float>& centroids_x, vector<float>& centroids_y) {
    int number_of_elements_in_a_cluster[k];
    float sum_x[k], sum_y[k];


    for (int iter = 0; iter < number_of_iterations; iter++) {

        for (int i = 0; i < n; i++) {
            float min_distance = FLT_MAX;
            for (int b = 0; b < k; b++) {
                float dist = (dataset.x[i] - centroids_x[b])*(dataset.x[i] - centroids_x[b]) + (dataset.y[i] - centroids_y[b])*(dataset.y[i] - centroids_y[b]);
                if (dist < min_distance) {
                    min_distance = dist;
                    dataset.cluster_id[i] = b;
                }
            }
        }

        memset(number_of_elements_in_a_cluster, 0, sizeof(int) * k);
        memset(sum_x, 0, sizeof(float) * k);
        memset(sum_y, 0, sizeof(float) * k);


        for (int i = 0; i < n; i++) {
            const int cluster_id = dataset.cluster_id[i];
            number_of_elements_in_a_cluster[cluster_id]++;
            sum_x[cluster_id] += dataset.x[i];
            sum_y[cluster_id] += dataset.y[i];

        }


        for (int i = 0; i < k; i++) {
            if (number_of_elements_in_a_cluster[i] > 0) {
                centroids_x[i] = sum_x[i] / (static_cast<float>(number_of_elements_in_a_cluster[i]));
                centroids_y[i] = sum_y[i] / (static_cast<float>(number_of_elements_in_a_cluster[i]));
            }
        }

    }
}
