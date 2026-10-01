#pragma once

#include <array>
#include <cassert>
#include <vector>

#include "../lin_alg/matrix.hpp"
#include "../lin_alg/vector.hpp"
#include "functions.hpp"
#include "parameters.hpp"

namespace neural {
    using namespace lin_alg;
    using std::string, std::pair, std::array, std::vector;

    template <class type>
    struct Network {
        using f_type = function<type,type>;

        protected:
        int _in_size, _out_size, _n_layer, _layer_size;
        f_type _activation;

        Parameters<type> params;

        const vector<vmatrix_t<type>>& weights() const {return params.weights();}
        const vector<vvect_t<type>>& biases() const {return params.biases();}

        public:
        const vmatrix_t<type>& get_weight(int layer) const {
            return params.get_weight(layer);
        }
        const vvect_t<type>& get_bias(int layer) const {
            return params.get_bias(layer);
        }
        void set_weight(int layer, const vmatrix_t<type>& m) {
            params.set_weight(layer, m);
        }
        void set_bias(int layer, const vvect_t<type>& v) {
            params.set_bias(layer, v);
        }

        type activation(type x) const {return _activation(x);}
        vvect_t<type> activation(const vvect_t<type>& x) const {
            vvect_t<type> val(x.size());
            for (int i = 0; i < x.size(); i++) {val[i] = activation(x[i]);}
            return val;
        }
        vmatrix_t<type> activation(const vmatrix_t<type>& m) const {
            vmatrix_t<type> val(m.nr(), m.nc());
            for (int i = 0; i < m.raw_size(); i++) {val.raw_at(i) = activation(m.raw_at(i));}
            return val;
        }
        void apply_activation(vvect_t<type>& x) const {
            for (int i = 0; i < x.size(); i++) {x[i] = activation(x[i]);}
        }

        Network(
            int in_size, int out_size, int n_layer, int layer_size,
            activation_t act = activation_t::SIGMOID
        ):
        _in_size(in_size), _out_size(out_size), _n_layer(n_layer), _layer_size(layer_size),
        _activation(get_activation<type>(act)),
        params(in_size, out_size, n_layer, layer_size) {
            assert(n_layer >= 3);
        }

        int in_size() const {return _in_size;}
        int out_size() const {return _out_size;}
        int n_layer() const {return _n_layer;}
        int layer_size() const {return _layer_size;}

        virtual vvect_t<type> compute(const vvect_t<type>& in) {
            assert(in.size() == in_size());
            vvect_t<type> out = in;
            for (int i = 0; i < n_layer(); i++) {
                out = out * weights()[i] + biases()[i];
                apply_activation(out);
            }
            return out;
        }
    };

    template <class type>
    struct StoredNetwork: Network<type> {
        protected:
        vector<vvect_t<type>> pre_activations;
        vector<vvect_t<type>> activations;

        public:
        StoredNetwork(
            int in_size, int out_size, int n_layer, int layer_size,
            activation_t act = activation_t::SIGMOID
        ):
        Network<type>(in_size, out_size, n_layer, layer_size, act) {
            pre_activations.reserve(n_layer);
            activations.reserve(n_layer);
            for (int i = 0; i < n_layer-1; i++) {
                pre_activations.push_back(vvect_t<type>(layer_size));
                activations.push_back(vvect_t<type>(layer_size));
            }
            pre_activations.push_back(vvect_t<type>(out_size));
            activations.push_back(vvect_t<type>(out_size));
        }

        vvect_t<type> compute(const vvect_t<type>& in) override {
            assert(in.size() == in_size());
            pre_activations[0] = in * weights()[0] + biases()[0];
            activations[0] = activation(pre_activations[0]);
            for (int i = 1; i < n_layer(); i++) {
                pre_activations[i] = activations[i-1] * weights()[i] + biases()[i];
                activations[i] = activation(pre_activations[i]);
            }
            return activations.back();
        }
    };
}