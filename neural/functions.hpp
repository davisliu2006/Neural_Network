#pragma once

#include <cmath>

#include "../lin_alg/vector.hpp"

namespace neural {
    using namespace lin_alg;

    template <class type>
    inline type sq(type x) {return x*x;}

    template <class type>
    inline type relu(type x) {
        return std::max(type(0), x);
    }

    template <class type>
    inline type sigmoid(type x) {
        return 1/(1 + std::exp(double(-x)));
    }
    template <class type>
    inline type sigmoid_d(type x) {
        double enx = std::exp(double(-x));
        return enx / sq(1+enx);
    }
}