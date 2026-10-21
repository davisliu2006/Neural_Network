#pragma once

#include "../../neural/neural.hpp"

using namespace lin_alg;
using namespace neural;

template <class type>
static int max_index(const vvect_t<type>& v) {
    int best = 0;
    for (int i = 1; i < v.size(); i++) {
        if (v[i] > v[best]) {best = i;}
    }
    return best;
}

template <class type>
static double accuracy(
    TrainableNetwork<type>& net,
    const vector<TrainingExample<type>>& examples
) {
    int correct = 0;
    for (const auto& [x, y]: examples) {
        if (max_index(get_row(net.compute(copy_row(x, 1)), 0)) == max_index(y)) {
            correct++;
        }
    }
    return double(correct)/double(examples.size());
}