#pragma once

#include <cmath>
#include <limits>

#include "../lin_alg/matrix_view.hpp"
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
        using StoredNetwork<type>::assign_activation;
        using StoredNetwork<type>::in_size, StoredNetwork<type>::out_size,
            StoredNetwork<type>::n_layer, StoredNetwork<type>::layer_size,
            StoredNetwork<type>::default_batch_size;
        using StoredNetwork<type>::compute;

        protected:
        f_type _activation_d, _last_activation_d;
        loss_t _loss;

        public:
        void assign_activation_d(
            vvect_t<type>& dst, const vvect_t<type>& x, bool last = false
        ) const {
            assert(dst.size() == x.size());
            auto f = (last? _last_activation_d : _activation_d);
            for (int i = 0; i < dst.size(); i++) {dst[i] = f(x[i]);}
        }
        void assign_activation_d(
            vmatrix_t<type>& dst, const vmatrix_t<type>& x, bool last = false
        ) const {
            assert(dst.nr() == x.nr() && dst.nc() == x.nc());
            auto f = (last? _last_activation_d : _activation_d);
            for (int i = 0; i < dst.raw_size(); i++) {dst.raw_at(i) = f(x.raw_at(i));}
        }

        TrainableNetwork(
            int in_size, int out_size, int n_layer, int layer_size,
            activation_t act = activation_t::SIGMOID
        ):
        TrainableNetwork(in_size, out_size, n_layer, layer_size, act, act) {}
        TrainableNetwork(
            int in_size, int out_size, int n_layer, int layer_size,
            activation_t act, activation_t last_act,
            loss_t loss = loss_t::MSE
        ):
        StoredNetwork<type>(in_size, out_size, n_layer, layer_size, act, last_act),
        _activation_d(get_activation_d<type>(act)),
        _last_activation_d(get_activation_d<type>(last_act)),
        _loss(loss) {
            if (loss == loss_t::CROSS_ENTROPY) {
                assert(last_act == activation_t::SIGMOID);
            }
        }

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
            int B = default_batch_size();
            for (int start = 0; start < int(examples.size()); start += B) {
                int remaining = int(examples.size()) - start;
                int n = remaining < B? remaining : B;
                auto [x, y] = make_batch(examples, start, n);
                if (_loss == loss_t::MSE) {
                    total += (compute(x) - y).vect_mag2();
                } else {
                    compute(x);
                    total += binary_cross_entropy(this->pre_activations.back(), y);
                }
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
        pair<vmatrix_t<type>, vmatrix_t<type>> make_batch(
            const vector<TrainingExample<type>>& examples, int start, int n
        ) const {
            vmatrix_t<type> x(n, in_size()), y(n, out_size());
            for (int i = 0; i < n; i++) {
                const auto& ex = examples[start+i];
                assert(ex.input.size() == in_size() && ex.output.size() == out_size());
                for (int j = 0; j < in_size(); j++) {x.at(i, j) = ex.input[j];}
                for (int j = 0; j < out_size(); j++) {y.at(i, j) = ex.output[j];}
            }
            return {x, y};
        }
        Parameters<type> compute_gradient(
            const vector<TrainingExample<type>>& examples
        ) {
            Parameters<type> grad(in_size(), out_size(), n_layer(), layer_size());
            Parameters<type> partial(in_size(), out_size(), n_layer(), layer_size());
            int B = default_batch_size();
            for (int start = 0; start < int(examples.size()); start += B) {
                int remaining = int(examples.size()) - start;
                int n = remaining < B? remaining : B;
                auto [x, y] = make_batch(examples, start, n);
                compute_partial_gradient(partial, x, y);
                grad += partial;
            }
            return grad/type(examples.size());
        }
        void compute_partial_gradient(
            Parameters<type>& grad,
            const vmatrix_t<type>& x,
            const vmatrix_t<type>& y
        ) {
            compute(x);
            // dE_dai = dE_dan
            vmatrix_t<type> dE_dai(
                this->activations.back().nr(), this->activations.back().nc()
            );
            assign_sub(dE_dai, this->activations.back(), y);
            if (_loss == loss_t::MSE) {assign_mul(dE_dai, dE_dai, type(2));}
            vmatrix_t<type> dE_dbi, dai_dbi;
            for (int i = n_layer()-1; i >= 0; i--) {
                const auto& a_prev = (
                    i == 0? this->normalized_in : this->activations[i-1]
                );
                const vmatrix_t<type>& w = weights()[i];
                const auto& z = this->pre_activations[i];
                if (dE_dbi.nr() != z.nr() || dE_dbi.nc() != z.nc()) {
                    dE_dbi = vmatrix_t<type>(z.nr(), z.nc());
                }
                if (dai_dbi.nr() != z.nr() || dai_dbi.nc() != z.nc()) {
                    dai_dbi = vmatrix_t<type>(z.nr(), z.nc());
                }
                if (_loss == loss_t::CROSS_ENTROPY && i == n_layer()-1) {
                    for (int k = 0; k < dai_dbi.raw_size(); k++) {
                        dai_dbi.raw_at(k) = type(1);
                    }
                } else {
                    assign_activation_d(dai_dbi, z, i == n_layer()-1);
                }
                assign_elem_prod(dE_dbi, dE_dai, dai_dbi);
                assign_sum_rows(grad.biases[i], dE_dbi);
                assign_mul(grad.weights[i], as_transpose(a_prev), dE_dbi);
                if (i > 0) {
                    if (dE_dai.nr() != dE_dbi.nr() || dE_dai.nc() != w.nr()) {
                        dE_dai = vmatrix_t<type>(dE_dbi.nr(), w.nr());
                    }
                    assign_mul(dE_dai, dE_dbi, as_transpose(w));
                }
            }
        }
    };
}