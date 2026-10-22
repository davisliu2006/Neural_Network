#pragma once

#include <cassert>
#include <vector>

#include "../lin_alg/matrix.hpp"
#include "../lin_alg/vector.hpp"

namespace neural {
    using std::vector;
    using namespace lin_alg;

    template <class type>
    struct Parameters {
        vector<vmatrix_t<type>> weights;
        vector<vvect_t<type>> biases;

        /*
        The following are safe versions of parameter access.
        */
        const vmatrix_t<type>& get_weight(int layer) const {
            assert(0 <= layer && layer < n_layer());
            return weights[layer];
        }
        const vvect_t<type>& get_bias(int layer) const {
            assert(0 <= layer && layer < n_layer());
            return biases[layer];
        }
        void set_weight(int layer, const vmatrix_t<type>& m) {
            assert(0 <= layer && layer < n_layer());
            assert(m.nr() == weights[layer].nr() && m.nc() == weights[layer].nc());
            weights[layer] = m;
        }
        void set_bias(int layer, const vvect_t<type>& v) {
            assert(0 <= layer && layer < n_layer());
            assert(v.size() == biases[layer].size());
            biases[layer] = v;
        }

        Parameters(int in_size, int out_size, int n_layer, int layer_size) {
            assert(n_layer >= 1);
            assert(in_size >= 1 && out_size >= 1 && layer_size >= 1);
            weights.reserve(n_layer);
            biases.reserve(n_layer);
            weights.push_back(vmatrix_t<type>(in_size, layer_size));
            biases.push_back(vvect_t<type>(layer_size));
            for (int i = 1; i < n_layer-1; i++) {
                weights.push_back(vmatrix_t<type>(layer_size, layer_size));
                biases.push_back(vvect_t<type>(layer_size));
            }
            weights.push_back(vmatrix_t<type>(layer_size, out_size));
            biases.push_back(vvect_t<type>(out_size));
        }
        Parameters(const vector<vmatrix_t<type>>& weights, const vector<vvect_t<type>>& biases):
        weights(weights), biases(biases) {
            assert(weights.size() == biases.size());
            assert(!weights.empty());
        }

        int n_layer() const {return int(weights.size());}

        void clear() {
            for (int i = 0; i < n_layer(); i++) {
                #pragma omp simd
                for (int k = 0; k < weights[i].raw_size(); k++) {
                    weights[i].raw_at(k) = type(0);
                }
                #pragma omp simd
                for (int k = 0; k < biases[i].size(); k++) {
                    biases[i][k] = type(0);
                }
            }
        }
    };

    // equals
    template <class type>
    inline bool operator ==(const Parameters<type>& p1, const Parameters<type>& p2) {
        assert(p1.n_layer() == p2.n_layer());
        for (int i = 0; i < p1.n_layer(); i++) {
            if (!(p1.weights[i] == p2.weights[i]) || !(p1.biases[i] == p2.biases[i])) {
                return false;
            }
        }
        return true;
    }

    // negative
    template <class type>
    inline Parameters<type> operator -(const Parameters<type>& p) {
        Parameters<type> val = p;
        for (int i = 0; i < val.n_layer(); i++) {
            apply_neg(val.weights[i]);
            apply_neg(val.biases[i]);
        }
        return val;
    }

    // addition
    template <class type>
    inline Parameters<type>& operator +=(Parameters<type>& p1, const Parameters<type>& p2) {
        assert(p1.n_layer() == p2.n_layer());
        for (int i = 0; i < p1.n_layer(); i++) {
            p1.weights[i] += p2.weights[i];
            p1.biases[i] += p2.biases[i];
        }
        return p1;
    }
    template <class type>
    inline Parameters<type> operator +(const Parameters<type>& p1, const Parameters<type>& p2) {
        Parameters<type> val = p1;
        val += p2;
        return val;
    }
    template <class type>
    inline Parameters<type>& operator -=(Parameters<type>& p1, const Parameters<type>& p2) {
        assert(p1.n_layer() == p2.n_layer());
        for (int i = 0; i < p1.n_layer(); i++) {
            p1.weights[i] -= p2.weights[i];
            p1.biases[i] -= p2.biases[i];
        }
        return p1;
    }
    template <class type>
    inline Parameters<type> operator -(const Parameters<type>& p1, const Parameters<type>& p2) {
        Parameters<type> val = p1;
        val -= p2;
        return val;
    }

    // scalar multiplication
    template <class type>
    inline Parameters<type>& operator *=(Parameters<type>& p, type scale) {
        for (int i = 0; i < p.n_layer(); i++) {
            p.weights[i] *= scale;
            p.biases[i] *= scale;
        }
        return p;
    }
    template <class type>
    inline Parameters<type> operator *(const Parameters<type>& p, type scale) {
        Parameters<type> val = p;
        val *= scale;
        return val;
    }
    template <class type>
    inline Parameters<type>& operator /=(Parameters<type>& p, type scale) {
        for (int i = 0; i < p.n_layer(); i++) {
            p.weights[i] /= scale;
            p.biases[i] /= scale;
        }
        return p;
    }
    template <class type>
    inline Parameters<type> operator /(const Parameters<type>& p, type scale) {
        Parameters<type> val = p;
        val /= scale;
        return val;
    }
}