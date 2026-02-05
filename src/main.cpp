#include <algorithm>
#include <ostream>
#include <iostream>
#include <chrono>
#include <random>
#include <stdio.h>

#include "Utility/read_dataset.h"
#include "Sequential Version/sequential_kmeans.h"
#include <time.h>
#ifdef PARALLEL
#include "Parallel Version/parallel_kmeans.h"
#include <omp.h>
#endif


#include "Utility/utils.h"

int main() {
    constexpr int k = 50;
    constexpr int number_of_iterations = 32;
    constexpr int n = 100000;

    Dataset_SoA dataset(n);
    read_csv("blobs.csv", n, dataset);
    constexpr int size = number_of_iterations - 2;
    vector<vector<float> > centroids_x(n, vector<float>(k, 0.0f));
    vector<vector<float> > centroids_y(n, vector<float>(k, 0.0f));
    float squared_distances[n] = {0.0f};
    vector<double> seq_wc_time(size, 0.0f);
    vector<double> seq_cpu_time(size, 0.0f);
    vector<float> sequential_centroids_x(k, 0.0f);
    vector<float> sequential_centroids_y(k, 0.0f);


#ifdef PARALLEL
    vector<double> par_wc_time(size, 0.0f);
    vector<double> par_cpu_time(size, 0.0f);
    vector<float> parallel_centroids_x(k, 0.0f);
    vector<float> parallel_centroids_y(k, 0.0f);

    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < number_of_iterations; i++) {
        mt19937_64 generator(1234);
        kmeansplusplus(dataset, k, generator, n, squared_distances, centroids_x[i].data(),
                       centroids_y[i].data());
    }
    auto end = chrono::high_resolution_clock::now();
    const chrono::duration<double, milli> duration = end - start;
    printf("Tempo necessario alla scelta dei %d centroidi iniziali (ms) : %.2f \n", k, duration.count());

    printf("Inizio esecuzione versione parallela... \n");

    for (int i = 0; i < number_of_iterations; i++) {
        parallel_centroids_x = centroids_x[i];
        parallel_centroids_y = centroids_y[i];
        clock_t start1 = clock();
        start = std::chrono::high_resolution_clock::now();
        parallel_kmeans(
            dataset, k, number_of_iterations, n, parallel_centroids_x.data(), parallel_centroids_y.data());
        end = std::chrono::high_resolution_clock::now();
        clock_t end1 = clock();
        const chrono::duration<double, milli> duration1 = end - start;
        double cpu_time = ((double) (end1 - start1)) / CLOCKS_PER_SEC;
        if (i > 1) {
            par_wc_time[i - 2] = duration1.count();
            par_cpu_time[i - 2] = cpu_time*1000;
        }
    }

    printf("Termine esecuzione versione parallela...\n");

    const double par_wc_mean = mean(par_wc_time);
    const double par_wc_dev_std = standard_dev(par_wc_time, par_wc_mean);
    const double min_wc_par_time = *min_element(par_wc_time.begin(), par_wc_time.end());
    const double max_wc_par_time = *max_element(par_wc_time.begin(), par_wc_time.end());

    const double par_cpu_mean = mean(par_cpu_time);
    const double par_cpu_dev_std = standard_dev(par_cpu_time, par_cpu_mean);
    const double min_cpu_par_time = *min_element(par_cpu_time.begin(), par_cpu_time.end());
    const double max_cpu_par_time = *max_element(par_cpu_time.begin(), par_cpu_time.end());


    printf("Versione parallela --> WC Time --> Min: %.2f, Max: %.2f, Mean: %.2f, Std_dev: %.2f\n", min_wc_par_time,
           max_wc_par_time, par_wc_mean, par_wc_dev_std);
    printf("Versione parallela --> CPU Time --> Min: %.2f, Max: %.2f, Mean: %.2f, Std_dev: %.2f\n", min_cpu_par_time,
           max_cpu_par_time, par_cpu_mean, par_cpu_dev_std);

    for (int i = 0; i < size; i++) {
        printf("par_wc_time[%d] = %.2f, par_cpu_time[%d] = %.2f\n", i, par_wc_time[i], i, par_cpu_time[i]);
    }

    for (int i = 0; i < k; i++) {
        printf("parallel_centroid_x [%d] = %.2f -- parallel_centroid_y [%d] = %.2f\n", i,
               parallel_centroids_x[i], i,
               parallel_centroids_y[i]);
    }
#else
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < number_of_iterations; i++) {
        mt19937_64 generator(1234);
        kmeansplusplus(dataset, k, generator, n, squared_distances, centroids_x[i].data(),
                       centroids_y[i].data());
    }
    auto end = chrono::high_resolution_clock::now();
    const chrono::duration<double, milli> duration = end - start;
    printf("Tempo necessario alla scelta dei %d centroidi iniziali (ms) : %.2f \n", k, duration.count());

    printf("Inizio esecuzione versione sequenziale... \n");

    for (int i = 0; i < number_of_iterations; i++) {
        sequential_centroids_x = centroids_x[i];
        sequential_centroids_y = centroids_y[i];
        clock_t start1 = clock();
        start = std::chrono::high_resolution_clock::now();
        sequential_kmeans(
            dataset, k, number_of_iterations, n, sequential_centroids_x.data(), sequential_centroids_y.data());
        end = std::chrono::high_resolution_clock::now();
        clock_t end1 = clock();
        const chrono::duration<double, milli> duration1 = end - start;
        double cpu_time = ((double) (end1 - start1)) / CLOCKS_PER_SEC;
        if (i > 1) {
            seq_wc_time[i - 2] = duration1.count();
            seq_cpu_time[i - 2] = cpu_time*1000;
        }
    }

        printf("Termine esecuzione versione sequenziale...\n");

        const double seq_wc_mean = mean(seq_wc_time);
        const double seq_wc_dev_std = standard_dev(seq_wc_time, seq_wc_mean);
        const double min_wc_seq_time = *min_element(seq_wc_time.begin(), seq_wc_time.end());
        const double max_wc_seq_time = *max_element(seq_wc_time.begin(), seq_wc_time.end());

        const double seq_cpu_mean = mean(seq_cpu_time);
        const double seq_cpu_dev_std = standard_dev(seq_cpu_time, seq_cpu_mean);
        const double min_cpu_seq_time = *min_element(seq_cpu_time.begin(), seq_cpu_time.end());
        const double max_cpu_seq_time = *max_element(seq_cpu_time.begin(), seq_cpu_time.end());

        printf("Versione sequenziale --> WC Time --> Min: %.2f, Max: %.2f, Mean: %.2f, Std_dev: %.2f\n", min_wc_seq_time, max_wc_seq_time,
               seq_wc_mean, seq_wc_dev_std);
        printf("Versione sequenziale --> CPU Time --> Min: %.2f, Max: %.2f, Mean: %.2f, Std_dev: %.2f\n", min_cpu_seq_time, max_cpu_seq_time,
               seq_cpu_mean, seq_cpu_dev_std);

        for (int i = 0; i < size; i++) {
            printf("seq_wc_time[%d] = %.2f, seq_cpu_time[%d] = %.2f\n", i, seq_wc_time[i],  i, seq_cpu_time[i]);
        }

        for (int i = 0; i < k; i++) {
            printf("sequential_centroid_x [%d] = %.2f -- sequential_centroid_y [%d] = %.2f\n", i,
                   sequential_centroids_x[i], i,
                   sequential_centroids_y[i]);
        }
#endif
    }


