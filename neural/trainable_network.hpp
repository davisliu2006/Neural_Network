#pragma once

#include <cmath>
#include <limits>

#include "network.hpp"
#include "rng.hpp"
#include "training_example.hpp"

namespace neural {
    template <class type>
    struct estimate_t {
        type value, lower, upper;
    };

    template <class type>
    struct TrainableNetwork: StoredNetwork<type> {
        static_assert(std::is_floating_point_v<type>);
        using f_type = typename Network<type>::f_type;
        using StoredNetwork<type>::weights, StoredNetwork<type>::biases;
        using StoredNetwork<type>::get_weight, StoredNetwork<type>::get_bias,
            StoredNetwork<type>::set_weight, StoredNetwork<type>::set_bias;
        using StoredNetwork<type>::activation, StoredNetwork<type>::apply_activation;
        using StoredNetwork<type>::in_size, StoredNetwork<type>::out_size,
            StoredNetwork<type>::n_layer, StoredNetwork<type>::layer_size;
        using StoredNetwork<type>::compute;

        protected:
        f_type _activation_d, _last_activation_d;

        public:
        vvect_t<type> activation_d(const vvect_t<type>& x, bool last = false) const {
            vvect_t<type> val = x;
            apply_activation_d(val, last);
            return val;
        }
        vvect_t<type>& apply_activation_d(vvect_t<type>& x, bool last = false) const {
            auto f = (last? _last_activation_d : _activation_d);
            for (int i = 0; i < x.size(); i++) {x[i] = f(x[i]);}
            return x;
        }

        TrainableNetwork(
            int in_size, int out_size, int n_layer, int layer_size,
            activation_t act = activation_t::SIGMOID
        ):
        TrainableNetwork(in_size, out_size, n_layer, layer_size, act, act) {}
        TrainableNetwork(
            int in_size, int out_size, int n_layer, int layer_size,
            activation_t act, activation_t last_act
        ):
        StoredNetwork<type>(in_size, out_size, n_layer, layer_size, act, last_act),
        _activation_d(get_activation_d<type>(act)),
        _last_activation_d(get_activation_d<type>(last_act)) {}

        void init_random() {
            for (int i = 0; i < n_layer(); i++) {
                vmatrix_t<type> w = get_weight(i);
                neural::init_random(
                    w, i == n_layer()-1? this->last_act() : this->act()
                );
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
            std::cout << "Training on " << examples.size() << " examples..." << std::endl;
            assert(!examples.empty());
            assert(learning_rate > 0);
            assert(epochs > 0);
            this->normalizer.init_from_examples(examples);
            vector<type> losses;
            losses.reserve(epochs);
            for (int epoch = 1; epoch <= epochs; epoch++) {
                std::cout << "Computing epoch " << epoch << "..." << std::endl;
                double t0 = time();
                this->params -= compute_gradient(examples)*learning_rate;
                double t1 = time();
                std::cout << "Epoch: " << epoch << " Loss: " << loss(examples)
                    << " in " << t1-t0 << "s" << std::endl;
                losses.push_back(loss(examples));
            }
            return losses;
        }

        /**
         * Geometrically probe a suggested learning rate.
         */
        estimate_t<type> probe_learning_rate(
            const vector<TrainingExample<type>>& examples,
            type lr_min, type lr_max,
            int n_trials, int probe_epochs
        ) {
            assert(!examples.empty());
            assert(0 < lr_min && lr_min < lr_max);
            assert(n_trials >= 2 && probe_epochs > 0);
            this->normalizer.init_from_examples(examples);
            Parameters<type> saved_params = this->params;

            vector<type> lrs(n_trials);
            type log_min = std::log(lr_min);
            type log_max = std::log(lr_max);
            type step = (log_max-log_min)/type(n_trials-1);
            for (int i = 0; i < n_trials; i++) {
                lrs[i] = std::exp(log_min + step*i);
            }

            int best_trial = 0;
            type best_loss = std::numeric_limits<type>::infinity();
            for (int trial = 0; trial < n_trials; trial++) {
                type lr = lrs[trial];
                std::cout << "Trying lr = " << lr << "... " << std::flush;
                this->params = saved_params;
                for (int epoch = 0; epoch < probe_epochs; epoch++) {
                    this->params -= compute_gradient(examples)*lr;
                }
                type curr_loss = loss(examples);
                std::cout << "Loss = " << curr_loss << std::endl;
                if (curr_loss < best_loss) {
                    best_trial = trial;
                    best_loss = curr_loss;
                }
            }
            this->params = saved_params;
            int lower = best_trial == 0? 0 : best_trial-1;
            int upper = best_trial == n_trials-1? best_trial : best_trial+1;
            return {lrs[best_trial], lrs[lower], lrs[upper]};
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
            // dE_dai = dE_dan
            vvect_t<type> dE_dai = (this->activations.back() - y)*type(2);
            for (int i = n_layer()-1; i >= 0; i--) {
                const auto& a_prev = (i == 0? this->normalized_in : this->activations[i-1]);
                // grad.bias[i] = dE_dai;
                grad.biases[i] = dE_dai;
                // grad.weight[i] = copy_row(dE_dai, weights()[i].nr());
                for (int j = 0; j < weights()[i].nr(); j++) {
                    #pragma omp simd
                    for (int k = 0; k < weights()[i].nc(); k++) {
                        grad.weights[i].at(j, k) = dE_dai[k];
                    }
                }
                vvect_t<type> dai_dbi = activation_d(
                    this->pre_activations[i], i == n_layer()-1
                );
                // grad.bias[i] = elem_prod(dE_dai, dai_dbi);
                apply_elem_prod(grad.biases[i], dai_dbi);
                // vmatrix_t<type> dai_dWi = elem_prod(
                //     copy_row(dai_dbi, weights()[i].nr()),
                //     copy_col(a_prev, weights()[i].nc())
                // );
                // grad.weight[i] = elem_prod(
                //     copy_row(dE_dai, weights()[i].nr()),
                //     dai_dWi
                // );
                for (int j = 0; j < weights()[i].nr(); j++) {
                    #pragma omp simd
                    for (int k = 0; k < weights()[i].nc(); k++) {
                        grad.weights[i].at(j, k) *= dai_dbi[k]*a_prev[j];
                    }
                }
                // dE_dai *= elem_prod(
                //     copy_row(dai_dbi, weights()[i].nr()),
                //     weights()[i]
                // ).transpose();
                vvect_t<type> dE_dai_new = vvect_t<type>(weights()[i].nr());
                for (int j = 0; j < weights()[i].nr(); j++) {
                    type sum = 0;
                    #pragma omp simd reduction(+:sum)
                    for (int k = 0; k < weights()[i].nc(); k++) {
                        sum += dE_dai[k]*dai_dbi[k]*weights()[i].at(j, k);
                    }
                    dE_dai_new[j] = sum;
                }
                dE_dai = dE_dai_new;
            }
        }
    };
}