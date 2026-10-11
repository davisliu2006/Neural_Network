#pragma once

#include "../lin_alg/vector.hpp"

namespace neural {
    using namespace lin_alg;

    template <class type>
    struct TrainingExample {
        vvect_t<type> input;
        vvect_t<type> output;

        TrainingExample(const vvect_t<type>& input, const vvect_t<type>& output):
        input(input), output(output) {}
    };
}
