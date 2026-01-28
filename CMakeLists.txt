cmake_minimum_required(VERSION 3.28)
project(kmeansclustering)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O3 -ftree-vectorize -march=native -mtune=native -ffast-math -flto")

add_executable(kmeansclustering src/main.cpp
        src/Utility/read_dataset.cpp
        src/Utility/read_dataset.h
        "src/Sequential Version/sequential_kmeans.cpp"
        "src/Sequential Version/sequential_kmeans.h"
        "src/Parallel Version/parallel_kmeans.cpp"
        "src/Parallel Version/parallel_kmeans.h"
        src/Utility/utils.cpp
        src/Utility/utils.h)

target_link_libraries(kmeansclustering)