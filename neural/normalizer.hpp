#pragma once

#include <vector>

#include "../lin_alg/matrix.hpp"
#include "../lin_alg/vector.hpp"
#include "base.hpp"
#include "training_example.hpp"

namespace neural {
    using namespace lin_alg;
    using std::vector;

    template <class type>
    struct Normalizer {
        type mean = 0, std_dev = 1;

        void assign_normalize(vvect_t<type>& dst, const vvect_t<type>& v) const {
            assert(dst.size() == v.size());
            for (int i = 0; i < dst.size(); i++) {
                dst[i] = (v[i] - mean) / std_dev;
            }
        }
        void assign_normalize(vmatrix_t<type>& dst, const vmatrix_t<type>& m) const {
            assert(dst.nr() == m.nr() && dst.nc() == m.nc());
            #pragma omp simd
            for (int i = 0; i < dst.raw_size(); i++) {
                dst.raw_at(i) = (m.raw_at(i) - mean) / std_dev;
            }
        }
        void init_from_examples(const vector<TrainingExample<type>>& examples) {
            mean = 0;
            int n = 0;
            for (const auto& example: examples) {
                for (type x: example.input) {mean += x;}
                n += example.input.size();
            }
            mean /= type(n);
            std_dev = 0;
            for (const auto& example: examples) {
                for (type x: example.input) {std_dev += sq(x-mean);}
            }
            std_dev = std::sqrt(std_dev/type(n));
        }
    };
}
