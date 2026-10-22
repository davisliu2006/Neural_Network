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

        void assign_activation(
            vvect_t<type>& dst, const vvect_t<type>& x, bool last = false
        ) const {
            assert(dst.size() == x.size());
            auto f = (last? _last_activation : _activation);
            for (int i = 0; i < dst.size(); i++) {dst[i] = f(x[i]);}
        }
        void assign_activation(
            vmatrix_t<type>& dst, const vmatrix_t<type>& x, bool last = false
        ) const {
            assert(dst.nr() == x.nr() && dst.nc() == x.nc());
            auto f = (last? _last_activation : _activation);
            for (int i = 0; i < dst.raw_size(); i++) {dst.raw_at(i) = f(x.raw_at(i));}
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

        virtual vmatrix_t<type> compute(const vmatrix_t<type>& in) {
            assert(in.nr() > 0 && in.nc() == in_size());
            vmatrix_t<type> out(in.nr(), in.nc());
            normalizer.assign_normalize(out, in);
            for (int i = 0; i < n_layer(); i++) {
                out = out * weights()[i];
                apply_add_row(out, biases()[i]);
                assign_activation(out, out, i == n_layer()-1);
            }
            return out;
        }
    };

    template <class type>
    struct StoredNetwork: Network<type> {
        using Network<type>::weights, Network<type>::biases;
        using Network<type>::get_weight, Network<type>::get_bias,
            Network<type>::set_weight, Network<type>::set_bias;
        using Network<type>::assign_activation;
        using Network<type>::in_size, Network<type>::out_size,
            Network<type>::n_layer, Network<type>::layer_size;

        protected:
        int _default_batch_size;
        vmatrix_t<type> normalized_in;
        vector<vmatrix_t<type>> pre_activations;
        vector<vmatrix_t<type>> activations;

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
        _default_batch_size(1),
        normalized_in(1, in_size) {
            pre_activations.reserve(n_layer);
            activations.reserve(n_layer);
            for (int i = 0; i < n_layer-1; i++) {
                pre_activations.push_back(vmatrix_t<type>(1, layer_size));
                activations.push_back(vmatrix_t<type>(1, layer_size));
            }
            pre_activations.push_back(vmatrix_t<type>(1, out_size));
            activations.push_back(vmatrix_t<type>(1, out_size));
        }

        int default_batch_size() const {return _default_batch_size;}
        void set_default_batch_size(int default_batch_size) {
            assert(default_batch_size > 0);
            _default_batch_size = default_batch_size;
            ensure_buffers(default_batch_size);
        }

        vmatrix_t<type> compute(const vmatrix_t<type>& in) override {
            assert(in.nr() > 0 && in.nc() == in_size());
            ensure_buffers(in.nr());
            this->normalizer.assign_normalize(normalized_in, in);
            assign_mul(pre_activations[0], normalized_in, weights()[0]);
            apply_add_row(pre_activations[0], biases()[0]);
            assign_activation(activations[0], pre_activations[0]);
            for (int i = 1; i < n_layer(); i++) {
                assign_mul(pre_activations[i], activations[i-1], weights()[i]);
                apply_add_row(pre_activations[i], biases()[i]);
                assign_activation(activations[i], pre_activations[i], i == n_layer()-1);
            }
            return activations.back();
        }

        protected:
        // Capacity is _default_batch_size: grow that if n exceeds it, then set
        // logical rows to n without shrinking storage.
        virtual void ensure_buffers(int n) {
            assert(n > 0);
            if (n > _default_batch_size) {_default_batch_size = n;}
            int cap = _default_batch_size;
            normalized_in.ensure_capacity(cap, in_size());
            normalized_in.set_shape(n, in_size());
            for (int i = 0; i < n_layer()-1; i++) {
                pre_activations[i].ensure_capacity(cap, layer_size());
                pre_activations[i].set_shape(n, layer_size());
                activations[i].ensure_capacity(cap, layer_size());
                activations[i].set_shape(n, layer_size());
            }
            pre_activations.back().ensure_capacity(cap, out_size());
            pre_activations.back().set_shape(n, out_size());
            activations.back().ensure_capacity(cap, out_size());
            activations.back().set_shape(n, out_size());
        }
    };
}