#include <algorithm>
#include <ostream>
#include <iostream>
#include <chrono>
#include <random>

#include "Utility/read_dataset.h"
#include "Sequential Version/sequential_kmeans.h"
#include "Parallel Version/parallel_kmeans.h"

#include "Utility/utils.h"

int main() {
    constexpr int k = 50;
    constexpr int number_of_iterations = 32;
    constexpr int n = 100000;

    Dataset_SoA dataset(n);

    read_csv("blobs.csv", n, dataset);



    vector<vector<float>> centroids_x(n, vector<float>(k, 0.0f));
    vector<vector<float>> centroids_y(n, vector<float>(k, 0.0f));

    float squared_distances[n] = {0.0f};



    vector<float> seq_time(number_of_iterations-2, 0.0f);
    vector<float> par_time(number_of_iterations-2, 0.0f);

    int size = number_of_iterations-2;

    vector<float> sequential_centroids_x(k, 0.0f);
    vector<float> sequential_centroids_y(k, 0.0f);
    vector<float> parallel_centroids_x(k, 0.0f);
    vector<float> parallel_centroids_y(k, 0.0f);

    int number_of_elements_in_a_cluster[k] = {0};
    float sum_x[k] = {0.0f};
    float sum_y[k] = {0.0f};

    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < number_of_iterations; i++) {
        mt19937_64 generator(std::random_device{}());
        kmeansplusplus(dataset, k, generator, n, squared_distances, centroids_x[i],
    centroids_y[i]);
    }
    auto end = chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("Tempo necessario alla scelta dei %d centroidi iniziali (ms) : %.2f \n", k, duration.count());

    printf("Inizio esecuzione versione sequenziale... \n");

    for (int i = 0; i < number_of_iterations; i++) {
        sequential_centroids_x = centroids_x[i];
        sequential_centroids_y = centroids_y[i];
        start = std::chrono::high_resolution_clock::now();
        sequential_kmeans(
            dataset, k, number_of_iterations, n, sequential_centroids_x, sequential_centroids_y, number_of_elements_in_a_cluster, sum_x, sum_y);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        if (i > 1) seq_time[i-2] = duration.count();
    }

    printf("Termine esecuzione versione sequenziale...\n");

    printf("Inizio esecuzione versione parallela... \n");

    for (int i = 0; i < number_of_iterations; i++) {
        parallel_centroids_x = centroids_x[i];
        parallel_centroids_y = centroids_y[i];
        start = std::chrono::high_resolution_clock::now();
        parallel_kmeans(
            dataset, k, number_of_iterations, n, parallel_centroids_x, parallel_centroids_y);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        if (i > 1) par_time[i-2] = duration.count();
    }

    printf("Termine esecuzione versione parallela...\n");

    printf("Inizio analisi dei tempi...\n");

    const float seq_mean = mean(seq_time);
    const float seq_dev_std = standard_dev(seq_time, seq_mean);
    const float min_seq_time = *min_element(seq_time.begin(), seq_time.end());
    const float max_seq_time = *max_element(seq_time.begin(), seq_time.end());

    const float par_mean = mean(par_time);
    const float par_dev_std = standard_dev(par_time, par_mean);
    const float min_par_time = *min_element(par_time.begin(), par_time.end());
    const float max_par_time = *max_element(par_time.begin(), par_time.end());

    printf("Versione sequenziale --> Min: %.2f, Max: %.2f, Mean: %.2f, Std_dev: %.2f\n", min_seq_time, max_seq_time, seq_mean, seq_dev_std);

    printf("Versione parallela --> Min: %.2f, Max: %.2f, Mean: %.2f, Std_dev: %.2f\n", min_par_time, max_par_time, par_mean, par_dev_std);

    for (int i = 0; i < size; i++) {
        printf("seq_time[%d] = %.2f, par_time[%d] = %.2f\n", i, seq_time[i], i, par_time[i]);
    }

printf("\n");


    for (int i = 0; i < k; i++) {
        printf("sequential_centroid_x [%d] = %.2f -- parallel_centroid_x [%d] = %.2f -- sequential_centroid_y [%d] = %.2f -- parallel_centroid_y [%d] = %.2f\n", i,
        sequential_centroids_x[i], i, parallel_centroids_x[i], i,
        sequential_centroids_y[i], i, parallel_centroids_y[i]);

    }
}
