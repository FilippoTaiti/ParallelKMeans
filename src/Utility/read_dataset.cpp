//
// Created by filippo on 15/01/26.
//

#include "read_dataset.h"

Dataset_SoA read_csv(const string &name, int n, Dataset_SoA &dataset) {
    ifstream file(name);

    if (!file.is_open()) {
        cerr << "Error opening dataset " << "blobs.csv" << endl;
        return dataset;
    }

    string line;

    int i = 0;
    while (getline(file, line) && i < n) {
        stringstream ss(line);
        string cell;

        getline(ss, cell, ',');
        dataset.x[i] = stof(cell);
        getline(ss, cell, ',');
        dataset.y[i] = stof(cell);

        getline(ss, cell, '\n');
        i++;
    }

    file.close();
    return dataset;
}
