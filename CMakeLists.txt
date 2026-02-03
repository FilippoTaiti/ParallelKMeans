cmake_minimum_required(VERSION 3.28)
project(kmeansclustering)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ftree-vectorize -march=native -mtune=native -ffast-math -flto")

find_package(OpenMP REQUIRED)

add_executable(kmeans_seq src/main.cpp
        src/Utility/read_dataset.cpp
        src/Utility/read_dataset.h
        "src/Sequential Version/sequential_kmeans.cpp"
        "src/Sequential Version/sequential_kmeans.h"
        src/Utility/utils.cpp
        src/Utility/utils.h)

add_executable(kmeans_par src/main.cpp
        src/Utility/read_dataset.cpp
        src/Utility/read_dataset.h
        "src/Parallel Version/parallel_kmeans.cpp"
        "src/Parallel Version/parallel_kmeans.h"
        src/Utility/utils.cpp
        src/Utility/utils.h)



target_link_libraries(kmeans_par PRIVATE OpenMP::OpenMP_CXX)
target_compile_definitions(kmeans_par PRIVATE PARALLEL)