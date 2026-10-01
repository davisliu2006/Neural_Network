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
        protected:
        vector<vmatrix_t<type>> _weights;
        vector<vvect_t<type>> _biases;

        public:
        const vector<vmatrix_t<type>>& weights() const {return _weights;}
        const vector<vvect_t<type>>& biases() const {return _biases;}

        const vmatrix_t<type>& get_weight(int layer) const {
            assert(0 <= layer && layer < n_layer());
            return _weights[layer];
        }
        const vvect_t<type>& get_bias(int layer) const {
            assert(0 <= layer && layer < n_layer());
            return _biases[layer];
        }
        void set_weight(int layer, const vmatrix_t<type>& m) {
            assert(0 <= layer && layer < n_layer());
            assert(m.nr() == _weights[layer].nr() && m.nc() == _weights[layer].nc());
            _weights[layer] = m;
        }
        void set_bias(int layer, const vvect_t<type>& v) {
            assert(0 <= layer && layer < n_layer());
            assert(v.size() == _biases[layer].size());
            _biases[layer] = v;
        }

        Parameters(int in_size, int out_size, int n_layer, int layer_size) {
            assert(n_layer >= 1);
            assert(in_size >= 1 && out_size >= 1 && layer_size >= 1);
            _weights.reserve(n_layer);
            _biases.reserve(n_layer);
            _weights.push_back(vmatrix_t<type>(in_size, layer_size));
            _biases.push_back(vvect_t<type>(layer_size));
            for (int i = 1; i < n_layer-1; i++) {
                _weights.push_back(vmatrix_t<type>(layer_size, layer_size));
                _biases.push_back(vvect_t<type>(layer_size));
            }
            _weights.push_back(vmatrix_t<type>(layer_size, out_size));
            _biases.push_back(vvect_t<type>(out_size));
        }
        Parameters(const vector<vmatrix_t<type>>& weights, const vector<vvect_t<type>>& biases):
        _weights(weights), _biases(biases) {
            assert(weights.size() == biases.size());
            assert(!weights.empty());
        }

        int n_layer() const {return int(_weights.size());}
    };

    // equals
    template <class type>
    inline bool operator ==(const Parameters<type>& p1, const Parameters<type>& p2) {
        assert(p1.n_layer() == p2.n_layer());
        for (int i = 0; i < p1.n_layer(); i++) {
            if (!(p1.get_weight(i) == p2.get_weight(i)) || !(p1.get_bias(i) == p2.get_bias(i))) {
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
            val.set_weight(i, -val.get_weight(i));
            val.set_bias(i, -val.get_bias(i));
        }
        return val;
    }

    // addition
    template <class type>
    inline Parameters<type> operator +(const Parameters<type>& p1, const Parameters<type>& p2) {
        Parameters<type> val = p1;
        val += p2;
        return val;
    }
    template <class type>
    inline Parameters<type>& operator +=(Parameters<type>& p1, const Parameters<type>& p2) {
        assert(p1.n_layer() == p2.n_layer());
        for (int i = 0; i < p1.n_layer(); i++) {
            p1.set_weight(i, p1.get_weight(i) + p2.get_weight(i));
            p1.set_bias(i, p1.get_bias(i) + p2.get_bias(i));
        }
        return p1;
    }
    template <class type>
    inline Parameters<type> operator -(const Parameters<type>& p1, const Parameters<type>& p2) {
        Parameters<type> val = p1;
        val -= p2;
        return val;
    }
    template <class type>
    inline Parameters<type>& operator -=(Parameters<type>& p1, const Parameters<type>& p2) {
        assert(p1.n_layer() == p2.n_layer());
        for (int i = 0; i < p1.n_layer(); i++) {
            p1.set_weight(i, p1.get_weight(i) - p2.get_weight(i));
            p1.set_bias(i, p1.get_bias(i) - p2.get_bias(i));
        }
        return p1;
    }

    // scalar multiplication
    template <class type>
    inline Parameters<type> operator *(const Parameters<type>& p, type scale) {
        Parameters<type> val = p;
        val *= scale;
        return val;
    }
    template <class type>
    inline Parameters<type>& operator *=(Parameters<type>& p, type scale) {
        for (int i = 0; i < p.n_layer(); i++) {
            p.set_weight(i, p.get_weight(i)*scale);
            p.set_bias(i, p.get_bias(i)*scale);
        }
        return p;
    }
    template <class type>
    inline Parameters<type> operator /(const Parameters<type>& p, type scale) {
        Parameters<type> val = p;
        val /= scale;
        return val;
    }
    template <class type>
    inline Parameters<type>& operator /=(Parameters<type>& p, type scale) {
        for (int i = 0; i < p.n_layer(); i++) {
            p.set_weight(i, p.get_weight(i)/scale);
            p.set_bias(i, p.get_bias(i)/scale);
        }
        return p;
    }
}