#pragma once

#include <cmath>

#include "../lin_alg/vector.hpp"
#include "base.hpp"

namespace neural {
    using namespace lin_alg;

    enum class activation_t {
        RELU,
        SIGMOID
    };

    template <class type>
    inline type relu(type x) {
        return std::max(type(0), x);
    }
    template <class type>
    inline type relu_d(type x) {
        return x > 0? type(1) : type(0);
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

    template <class type>
    inline function<type,type> get_activation(activation_t act) {
        switch (act) {
            case activation_t::RELU: {return relu<type>;}
            case activation_t::SIGMOID: {return sigmoid<type>;}
        }
    }
    template <class type>
    inline function<type,type> get_activation_d(activation_t act) {
        switch (act) {
            case activation_t::RELU: {return relu_d<type>;}
            case activation_t::SIGMOID: {return sigmoid_d<type>;}
        }
    }
}