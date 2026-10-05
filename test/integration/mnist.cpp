#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "../test.hpp"
#include "../../neural/io/idx.hpp"

using namespace std;
using namespace lin_alg;
using namespace neural;

static bool mnist_present() {
    ifstream images("data/mnist/train-images-idx3-ubyte", ios::binary);
    ifstream labels("data/mnist/train-labels-idx1-ubyte", ios::binary);
    return bool(images) && bool(labels);
}

static int max_index(const vvect_t<double>& v) {
    int best = 0;
    for (int i = 1; i < v.size(); i++) {
        if (v[i] > v[best]) {best = i;}
    }
    return best;
}

static double accuracy(
    TrainableNetwork<double>& net,
    const vector<TrainingExample<double>>& examples
) {
    int correct = 0;
    for (const auto& [x, y]: examples) {
        if (max_index(net.compute(x)) == max_index(y)) {correct++;}
    }
    return double(correct)/double(examples.size());
}

TestBatch tests({
    {"mnist subset train", []() {
        auto examples = idx::load_mnist<double>(
            "data/mnist/train-images-idx3-ubyte",
            "data/mnist/train-labels-idx1-ubyte"
        );
        // full dataset is slow, start with a subset
        const int subset = 200;
        assert(int(examples.size()) >= subset);
        examples.erase(examples.begin() + subset, examples.end());

        TrainableNetwork<double> net(784, 10, 3, 64, activation_t::SIGMOID);
        net.init_random();

        const double learning_rate = 1.5;
        const int epochs = 50;
        double t0 = time();
        vector<double> losses = net.train(examples, learning_rate, epochs);
        double t1 = time();
        cout << "    training time: " << t1-t0 << "s" << "\n";

        cout << "    initial loss: " << losses.front() << "\n";
        cout << "    final loss:   " << losses.back() << "\n";
        double acc = accuracy(net, examples);
        cout << "    accuracy:     " << acc << "\n";

        assert(losses.back() < losses.front());
        assert(acc > 0.7);
    }}
});

int main() {
    ios::sync_with_stdio(false);
    cin.tie(NULL);
    cout << setprecision(7);

    if (!mnist_present()) {
        cout << "Skip: MNIST IDX files not found under data/mnist/\n";
        return 0;
    }
    tests.run();

    return 0;
}
