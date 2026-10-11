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
    {"mnist subset train", []() {
        assert(dataset.size() >= 300);
        auto examples = vector<TrainingExample<double>>(
            dataset.begin(), dataset.begin() + 200
        );
        auto test_examples = vector<TrainingExample<double>>(
            dataset.begin() + 200, dataset.begin() + 300
        );
        
        TrainableNetwork<double> net(784, 10, 3, 64, activation_t::SIGMOID);
        net.init_random();

        const double learning_rate = 1;
        const int epochs = 50;
        double t0 = time();
        vector<double> losses = net.train(examples, learning_rate, epochs);
        double t1 = time();
        cout << "Training time: " << t1-t0 << "s" << "\n";

        cout << "Initial loss: " << losses.front() << "\n";
        cout << "Final loss:   " << losses.back() << "\n";
        double acc = accuracy(net, examples);
        cout << "Accuracy:     " << acc << "\n";

        double acc_new = accuracy(net, test_examples);
        cout << "New data accuracy:" << acc_new << "\n";

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