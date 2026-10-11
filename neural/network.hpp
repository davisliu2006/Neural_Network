#pragma once

#include <array>
#include <cassert>
#include <vector>

#include "../lin_alg/matrix.hpp"
#include "../lin_alg/vector.hpp"
#include "functions.hpp"
#include "normalizer.hpp"
#include "parameters.hpp"

namespace neural {
    using namespace lin_alg;
    using std::string, std::pair, std::array, std::vector;

    template <class type>
    struct Network {
        using f_type = function<type,type>;

        protected:
        int _in_size, _out_size, _n_layer, _layer_size;
        activation_t _act, _last_act;
        f_type _activation, _last_activation;

        Parameters<type> params;
        Normalizer<type> normalizer;

        const vector<vmatrix_t<type>>& weights() const {return params.weights;}
        const vector<vvect_t<type>>& biases() const {return params.biases;}

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

        vvect_t<type> activation(const vvect_t<type>& x, bool last = false) const {
            vvect_t<type> val = x;
            apply_activation(val, last);
            return val;
        }
        vvect_t<type>& apply_activation(vvect_t<type>& x, bool last = false) const {
            auto f = (last? _last_activation : _activation);
            for (int i = 0; i < x.size(); i++) {x[i] = f(x[i]);}
            return x;
        }

        Network(
            int in_size, int out_size, int n_layer, int layer_size,
            activation_t act = activation_t::SIGMOID
        ):
        Network(in_size, out_size, n_layer, layer_size, act, act) {}
        Network(
            int in_size, int out_size, int n_layer, int layer_size,
            activation_t act, activation_t last_act
        ):
        _in_size(in_size), _out_size(out_size), _n_layer(n_layer), _layer_size(layer_size),
        _act(act), _last_act(last_act),
        _activation(get_activation<type>(act)),
        _last_activation(get_activation<type>(last_act)),
        params(in_size, out_size, n_layer, layer_size) {
            assert(n_layer >= 3);
        }

        int in_size() const {return _in_size;}
        int out_size() const {return _out_size;}
        int n_layer() const {return _n_layer;}
        int layer_size() const {return _layer_size;}
        activation_t act() const {return _act;}
        activation_t last_act() const {return _last_act;}

        virtual vvect_t<type> compute(const vvect_t<type>& in) {
            assert(in.size() == in_size());
            vvect_t<type> out = normalizer.normalize(in);
            for (int i = 0; i < n_layer(); i++) {
                out = out * weights()[i] + biases()[i];
                apply_activation(out, i == n_layer()-1);
            }
            return out;
        }
    };

    template <class type>
    struct StoredNetwork: Network<type> {
        using Network<type>::weights, Network<type>::biases;
        using Network<type>::get_weight, Network<type>::get_bias,
            Network<type>::set_weight, Network<type>::set_bias;
        using Network<type>::activation, Network<type>::apply_activation;
        using Network<type>::in_size, Network<type>::out_size,
            Network<type>::n_layer, Network<type>::layer_size;

        protected:
        vvect_t<type> normalized_in;
        vector<vvect_t<type>> pre_activations;
        vector<vvect_t<type>> activations;

        public:
        StoredNetwork(
            int in_size, int out_size, int n_layer, int layer_size,
            activation_t act = activation_t::SIGMOID
        ):
        StoredNetwork(in_size, out_size, n_layer, layer_size, act, act) {}
        StoredNetwork(
            int in_size, int out_size, int n_layer, int layer_size,
            activation_t act, activation_t last_act
        ):
        Network<type>(in_size, out_size, n_layer, layer_size, act, last_act),
        normalized_in(in_size) {
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
            normalized_in = this->normalizer.normalize(in);
            pre_activations[0] = normalized_in * weights()[0] + biases()[0];
            activations[0] = activation(pre_activations[0]);
            for (int i = 1; i < n_layer(); i++) {
                pre_activations[i] = activations[i-1] * weights()[i] + biases()[i];
                activations[i] = activation(pre_activations[i], i == n_layer()-1);
            }
            return activations.back();
        }
    };
}