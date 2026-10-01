#pragma once

#include "network.hpp"
#include "rng.hpp"

namespace neural {
    template <class type>
    struct TrainingExample {
        vvect_t<type> input;
        vvect_t<type> output;

        TrainingExample(const vvect_t<type>& input, const vvect_t<type>& output):
        input(input), output(output) {}
    };

    template <class type>
    struct TrainableNetwork: StoredNetwork<type> {
        static_assert(std::is_floating_point_v<type>);
        using f_type = typename Network<type>::f_type;

        protected:
        f_type _activation_d;

        public:
        type activation_d(type x) const {return _activation_d(x);}
        vvect_t<type> activation_d(const vvect_t<type>& x) const {
            vvect_t<type> val(x.size());
            for (int i = 0; i < x.size(); i++) {val[i] = activation_d(x[i]);}
            return val;
        }
        vmatrix_t<type> activation_d(const vmatrix_t<type>& m) const {
            vmatrix_t<type> val(m.nr(), m.nc());
            for (int i = 0; i < m.raw_size(); i++) {val.raw_at(i) = activation_d(m.raw_at(i));}
            return val;
        }
        void apply_activation_d(vvect_t<type>& x) const {
            for (int i = 0; i < x.size(); i++) {x[i] = activation_d(x[i]);}
        }

        TrainableNetwork(
            int in_size, int out_size, int n_layer, int layer_size,
            activation_t act = activation_t::SIGMOID
        ):
        StoredNetwork<type>(in_size, out_size, n_layer, layer_size, act),
        _activation_d(get_activation_d<type>(act)) {}

        void init_random() {
            for (int i = 0; i < n_layer(); i++) {
                vmatrix_t<type> w = get_weight(i);
                neural::init_random(w);
                set_weight(i, w);
            }
        }

        type loss(const vector<TrainingExample<type>>& examples) {
            assert(!examples.empty());
            type total = 0;
            for (const auto& [x, y]: examples) {
                total += (compute(x) - y).mag2();
            }
            return total/type(examples.size());
        }

        vector<type> train(
            const vector<TrainingExample<type>>& examples,
            type learning_rate,
            int epochs
        ) {
            assert(!examples.empty());
            assert(learning_rate > 0);
            assert(epochs > 0);
            vector<type> losses;
            losses.reserve(epochs);
            while (epochs-- > 0) {
                this->params -= compute_gradient(examples)*learning_rate;
                losses.push_back(loss(examples));
            }
            return losses;
        }

        protected:
        // HELPERS
        Parameters<type> compute_gradient(
            const vector<TrainingExample<type>>& examples
        ) {
            Parameters<type> grad(in_size(), out_size(), n_layer(), layer_size());
            Parameters<type> partial(in_size(), out_size(), n_layer(), layer_size());
            for (const auto& example: examples) {
                compute_partial_gradient(partial, example);
                grad += partial;
            }
            return grad/type(examples.size());
        }
        void compute_partial_gradient(
            Parameters<type>& grad,
            const TrainingExample<type>& example
        ) {
            const auto& [x, y] = example;
            compute(x);
            vvect_t<type> dE_dan = (this->activations.back() - y)*type(2);
            vmatrix_t<type> dan_dai = vmatrix_t<type>::ID(out_size());
            for (int i = n_layer()-1; i >= 0; i--) {
                const auto& a_prev = (i == 0? x : this->activations[i-1]);
                vvect_t<type> dai_dbi = activation_d(this->pre_activations[i]);
                vmatrix_t<type> dai_dWi = elem_prod(
                    copy_row(dai_dbi, weights()[i].nr()),
                    copy_col(a_prev, weights()[i].nc())
                );
                grad.set_bias(i, elem_prod(dE_dan*dan_dai, dai_dbi));
                grad.set_weight(i, elem_prod(
                    copy_row(dE_dan*dan_dai, weights()[i].nr()),
                    dai_dWi
                ));
                dan_dai *= elem_prod(
                    copy_row(dai_dbi, weights()[i].nr()),
                    weights()[i]
                ).transpose();
            }
        }
    };
}