#include <cassert>
#include <iomanip>
#include <iostream>

#include "../test.hpp"
#include "../../neural/neural.hpp"
#include "helpers.hpp"

using namespace std;
using namespace lin_alg;
using namespace neural;

vector<TrainingExample<double>> dataset = idx::load_mnist<double>(
    "data/mnist/train-images-idx3-ubyte",
    "data/mnist/train-labels-idx1-ubyte"
);

TestBatch tests({
    {"mnist estimate learning rate", []() {
        assert(dataset.size() >= 200);
        auto examples = vector<TrainingExample<double>>(
            dataset.begin(), dataset.begin() + 200
        );
        auto lr_examples = vector<TrainingExample<double>>(
            dataset.begin(), dataset.begin() + 20
        );

        TrainableNetwork<double> net(784, 10, 3, 64, activation_t::SIGMOID);
        net.init_random();

        estimate_t<double> lr_estimate = {NAN, 0.5, 2};
        for (int i = 0; i < 2; i++) {
            lr_estimate = net.probe_learning_rate(
                lr_examples, lr_estimate.lower, lr_estimate.upper, 10, 20
            );
        }
        cout << "Estimated learning rate: " << lr_estimate.value
            << " in range [" << lr_estimate.lower << ", " << lr_estimate.upper << "]\n";
        const int epochs = 50;
        double t0 = time();
        vector<double> losses = net.train(examples, lr_estimate.value, epochs);
        double t1 = time();
        cout << "    training time: " << t1-t0 << "s" << "\n";

        cout << "    initial loss: " << losses.front() << "\n";
        cout << "    final loss:   " << losses.back() << "\n";
        double acc = accuracy(net, examples);
        cout << "    accuracy:     " << acc << "\n";

        assert(losses.back() < losses.front());
        assert(acc >= 0.7);
    }}
});

int main() {
    ios::sync_with_stdio(false);
    cin.tie(NULL);
    cout << setprecision(7);

    tests.run();

    return 0;
}