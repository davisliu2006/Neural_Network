#pragma once

#include <algorithm>
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
        // internal buffers
        Parameters<type> grad, partial;
        vmatrix_t<type> batch_x, batch_y;
        vmatrix_t<type> dE_dai, dE_dbi;

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
        _loss(loss),
        grad(in_size, out_size, n_layer, layer_size),
        partial(in_size, out_size, n_layer, layer_size) {
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
                make_batch(examples, start, n);
                if (_loss == loss_t::MSE) {
                    total += (compute(batch_x) - batch_y).vect_mag2();
                } else {
                    compute(batch_x);
                    total += binary_cross_entropy(
                        this->pre_activations.back(), batch_y
                    );
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
                compute_gradient(examples);
                grad *= learning_rate;
                this->params -= grad;
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
                    compute_gradient(examples);
                    grad *= lr;
                    this->params -= grad;
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
        void ensure_buffers(int n) override {
            StoredNetwork<type>::ensure_buffers(n);
            int cap = this->_default_batch_size;
            int W = std::max(layer_size(), out_size());
            batch_x.ensure_capacity(cap, in_size());
            batch_y.ensure_capacity(cap, out_size());
            dE_dai.ensure_capacity(cap, W);
            dE_dbi.ensure_capacity(cap, W);
        }

        // HELPERS
        void make_batch(
            const vector<TrainingExample<type>>& examples, int start, int n
        ) {
            batch_x.set_shape(n, in_size());
            batch_y.set_shape(n, out_size());
            for (int i = 0; i < n; i++) {
                const auto& ex = examples[start+i];
                assert(ex.input.size() == in_size() && ex.output.size() == out_size());
                #pragma omp simd
                for (int j = 0; j < in_size(); j++) {batch_x.at(i, j) = ex.input[j];}
                #pragma omp simd
                for (int j = 0; j < out_size(); j++) {batch_y.at(i, j) = ex.output[j];}
            }
        }
        void compute_gradient(const vector<TrainingExample<type>>& examples) {
            grad.clear();
            int B = default_batch_size();
            for (int start = 0; start < int(examples.size()); start += B) {
                int remaining = int(examples.size()) - start;
                int n = remaining < B? remaining : B;
                make_batch(examples, start, n);
                compute_partial_gradient();
                grad += partial;
            }
            grad /= type(examples.size());
        }
        void compute_partial_gradient() {
            compute(batch_x);
            int n = batch_x.nr();
            dE_dai.set_shape(n, out_size());
            // dE_dai = dE_dan
            assign_sub(dE_dai, this->activations.back(), batch_y);
            if (_loss == loss_t::MSE) {assign_mul(dE_dai, dE_dai, type(2));}
            for (int i = n_layer()-1; i >= 0; i--) {
                const auto& a_prev = (
                    i == 0? this->normalized_in : this->activations[i-1]
                );
                const vmatrix_t<type>& w = weights()[i];
                const auto& z = this->pre_activations[i];
                dE_dbi.set_shape(n, z.nc());
                if (_loss == loss_t::CROSS_ENTROPY && i == n_layer()-1) {
                    assign_elems(dE_dbi, type(1));
                } else {
                    assign_activation_d(dE_dbi, z, i == n_layer()-1);
                }
                apply_elem_prod(dE_dbi, dE_dai);
                assign_sum_rows(partial.biases[i], dE_dbi);
                assign_mul(partial.weights[i], as_transpose(a_prev), dE_dbi);
                if (i > 0) {
                    dE_dai.set_shape(n, w.nr());
                    assign_mul(dE_dai, dE_dbi, as_transpose(w));
                }
            }
        }
    };
}