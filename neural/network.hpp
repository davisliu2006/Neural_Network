#pragma once

#include <array>
#include <cassert>
#include <vector>

#include "../lin_alg/matrix.hpp"
#include "../lin_alg/vector.hpp"
#include "functions.hpp"

namespace neural {
    using namespace lin_alg;
    using std::string, std::pair, std::array, std::vector;

    template <class type>
    struct Network {
        using f_type = type (*)(type);
        
        protected:
        int _in_size, _out_size, _n_layer, _layer_size;
        f_type _smooth_f;

        vector<vmatrix_t<type>> transforms;
        vector<vvect_t<type>> biases;

        public:
        const vmatrix_t<type>& get_transform(int layer) const {
            assert(0 <= layer && layer < n_layer());
            return transforms[layer];
        }
        const vvect_t<type>& get_bias(int layer) const {
            assert(0 <= layer && layer < n_layer());
            return biases[layer];
        }
        void set_transform(int layer, const vmatrix_t<type>& m) {
            assert(0 <= layer && layer < n_layer());
            assert(m.nr() == transforms[layer].nr() && m.nc() == transforms[layer].nc());
            transforms[layer] = m;
        }
        void set_bias(int layer, const vvect_t<type>& v) {
            assert(0 <= layer && layer < n_layer());
            assert(v.size() == biases[layer].size());
            biases[layer] = v;
        }

        type smooth_f(type x) const {return _smooth_f(x);}
        vvect_t<type> smooth_f(const vvect_t<type>& x) const {
            vvect_t<type> val(x.size());
            for (int i = 0; i < x.size(); i++) {val[i] = smooth_f(x[i]);}
            return val;
        }
        void apply_smooth_f(vvect_t<type>& x) const {
            for (int i = 0; i < x.size(); i++) {x[i] = smooth_f(x[i]);}
        }

        Network(
            int in_size, int out_size, int n_layer, int layer_size,
            f_type smooth_f = sigmoid<type>
        ):
        _in_size(in_size), _out_size(out_size), _n_layer(n_layer), _layer_size(layer_size),
        _smooth_f(smooth_f) {
            assert(n_layer >= 3);
            assert(in_size >= 1 && out_size >= 1 && layer_size >= 1);
            transforms.push_back(vmatrix_t<type>(in_size, layer_size));
            biases.push_back(vvect_t<type>(layer_size));
            for (int i = 1; i < n_layer-1; i++) {
                transforms.push_back(vmatrix_t<type>(layer_size, layer_size));
                biases.push_back(vvect_t<type>(layer_size));
            }
            transforms.push_back(vmatrix_t<type>(layer_size, out_size));
            biases.push_back(vvect_t<type>(out_size));
        }

        int in_size() const {return _in_size;}
        int out_size() const {return _out_size;}
        int n_layer() const {return _n_layer;}
        int layer_size() const {return _layer_size;}

        vvect_t<type> compute(const vvect_t<type>& in) const {
            assert(in.size() == in_size());
            vvect_t<type> out = in;
            for (int i = 0; i < n_layer(); i++) {
                out = out * transforms[i] + biases[i];
                apply_smooth_f(out);
            }
            return out;
        }
    };
}