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

        vvect_t<type> normalize(const vvect_t<type>& v) const {
            vvect_t<type> val = v;
            return apply_normalize(val);
        }
        vvect_t<type>& apply_normalize(vvect_t<type>& v) const {
            for (type& x: v) {(x -= mean) /= std_dev;}
            return v;
        }
        vmatrix_t<type> normalize(const vmatrix_t<type>& m) const {
            vmatrix_t<type> val = m;
            return apply_normalize(val);
        }
        vmatrix_t<type>& apply_normalize(vmatrix_t<type>& m) const {
            #pragma omp simd
            for (int i = 0; i < m.raw_size(); i++) {
                (m.raw_at(i) -= mean) /= std_dev;
            }
            return m;
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