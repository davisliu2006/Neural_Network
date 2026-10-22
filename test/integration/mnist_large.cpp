#include <cassert>
#include <iomanip>
#include <iostream>

#include "../test.hpp"
#include "../../neural/neural.hpp"
#include "helpers.hpp"

using namespace std;
using namespace lin_alg;
using namespace neural;

vector<TrainingExample<float>> dataset = idx::load_mnist<float>(
    "data/mnist/train-images-idx3-ubyte",
    "data/mnist/train-labels-idx1-ubyte"
);

TestBatch tests({
    {"mnist subset train", []() {
        assert(dataset.size() >= 1200);
        auto examples = vector<TrainingExample<float>>(
            dataset.begin(), dataset.begin() + 3000
        );
        auto lr_examples = vector<TrainingExample<float>>(
            dataset.begin(), dataset.begin() + 20
        );
        auto test_examples = vector<TrainingExample<float>>(
            dataset.begin() + 3000, dataset.begin() + 4000
        );
        
        TrainableNetwork<float> net(784, 10, 3, 64, activation_t::RELU, activation_t::SIGMOID);
        net.set_default_batch_size(64);
        net.init_random();

        estimate_t<float> lr_estimate = {NAN, 0.05, 0.5};
        for (int i = 0; i < 2; i++) {
            lr_estimate = net.probe_learning_rate(
                lr_examples, lr_estimate.lower, lr_estimate.upper, 10, 20
            );
        }
        cout << "Estimated learning rate: " << lr_estimate.value
            << " in range [" << lr_estimate.lower << ", " << lr_estimate.upper << "]\n";
        const int epochs = 100;
        double t0 = time();
        vector<float> losses = net.train(examples, lr_estimate.value, epochs);
        double t1 = time();
        cout << "Training time: " << t1-t0 << "s" << "\n";

        cout << "Initial loss: " << losses.front() << "\n";
        cout << "Final loss:   " << losses.back() << "\n";
        double acc = accuracy(net, examples);
        cout << "Accuracy:     " << acc << "\n";

        double acc_new = accuracy(net, test_examples);
        cout << "New data accuracy:" << acc_new << "\n";

        assert(losses.back() < losses.front());
        assert(acc >= 0.93);
        assert(acc_new >= 0.9);
    }}
});

int main() {
    ios::sync_with_stdio(false);
    cin.tie(NULL);
    cout << setprecision(7);

    tests.run();

    return 0;
}