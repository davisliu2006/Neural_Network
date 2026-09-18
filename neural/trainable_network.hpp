#pragma once

#include "network.hpp"
#include "rng.hpp"

namespace neural {
    template <class type>
    struct TrainableNetwork: Network<type> {
        static_assert(std::is_floating_point_v<type>);
        using f_type = typename Network<type>::f_type;

        f_type _smooth_f_d;

        TrainableNetwork(
            int in_size, int out_size, int n_layer, int layer_size,
            f_type smooth_f = sigmoid<type>, f_type smooth_f_d = sigmoid_d<type>
        ):
        Network<type>(in_size, out_size, n_layer, layer_size, smooth_f),
        _smooth_f_d(smooth_f_d) {}

        void init_random() {
            for (auto& transform: this->transforms) {
                neural::init_random(transform);
            }
        }
    };
}