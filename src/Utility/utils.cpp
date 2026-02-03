//
// Created by filippo on 15/01/26.
//

#include "utils.h"


void kmeansplusplus(const Dataset_SoA &dataset, const int k, mt19937_64 &generator, const int n, float* __restrict squared_distances, vector<float>& centroids_x,
    vector<float>& centroids_y) {

    uniform_int_distribution<> dis(1, n);
    const int random_index = dis(generator);
    centroids_x[0] = (dataset.x[random_index]);
    centroids_y[0] = (dataset.y[random_index]);



    for (int l = 1; l < k ; l++) {
        float sum = 0.0f;

        for (int i = 0; i < n; i++) {
            float min_dist = FLT_MAX;
            for (int b = 0; b < l; b++) {
                float dist = 0;
                float diff = dataset.x[i] - centroids_x[b];
                diff += dataset.y[i] - centroids_y[b];
                dist += diff * diff;
                if (dist < min_dist)
                    min_dist = dist;
            }
            squared_distances[i] = min_dist;
            sum += min_dist;
        }

        uniform_real_distribution<> dis2(0, sum);
        const float threshold = dis2(generator);
        float tot = 0;
        for (size_t i = 0; i < n; i++) {
            tot += squared_distances[i];
            if (tot > threshold) {
                centroids_x[l] = (dataset.x[i]);
                centroids_y[l] = (dataset.y[i]);
                break;
            }
        }
    }
}


float mean(const vector<float> &vector) {
    float sum = 0.0f;
    const int N = vector.size();
    for (int i = 0; i < N; i++) {
        sum += vector[i];
    }
    return sum / static_cast<float>(vector.size());
}

float standard_dev(const vector<float> &vector, const float mean) {
    float sum = 0.0f;
    int N = vector.size();
    for (int i = 0; i < N; i++) {
        sum += (vector[i] - mean) * (vector[i] - mean);
    }

    return sqrt(sum / static_cast<float>(vector.size()));
}

bool isEqual(float a, float b) {
    return fabsf(a-b) <= FLT_EPSILON * fmaxf(1.0f, fmaxf(fabsf(a), fabsf(b)));

}