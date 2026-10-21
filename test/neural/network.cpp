#include <cassert>
#include <iomanip>
#include <iostream>

#include "../test.hpp"
#include "../../neural/network.hpp"

using namespace std;
using namespace lin_alg;
using namespace neural;

TestBatch tests({
    {"basic", []() {
        Network<int> net(3, 2, 4, 5, activation_t::RELU);
        assert(net.in_size() == 3 && net.out_size() == 2);
        assert(net.n_layer() == 4 && net.layer_size() == 5);
        // first layer size
        assert(net.get_weight(0).nr() == 3 && net.get_weight(0).nc() == 5);
        assert(net.get_bias(0).size() == 5);
        // middle layer sizes
        for (int i = 1; i < net.n_layer()-1; i++) {
            assert(net.get_weight(i).nr() == 5 && net.get_weight(i).nc() == 5);
            assert(net.get_bias(i).size() == 5);
        }
        // last layer size
        assert(net.get_weight(3).nr() == 5 && net.get_weight(3).nc() == 2);
        assert(net.get_bias(3).size() == 2);
        // compute
        assert(net.compute(copy_row(vvect_t<int>({1, 2, 3}), 1)) == vmatrix_t<int>(1, 2));
    }},
    {"compute identity network", []() {
        Network<int> net(3, 3, 3, 3, activation_t::RELU);
        net.set_weight(0, vmatrix_t<int>::ID(3));
        net.set_weight(1, vmatrix_t<int>::ID(3));
        net.set_weight(2, vmatrix_t<int>::ID(3));
        assert(
            net.compute(copy_row(vvect_t<int>({1, 2, 3}), 1))
            == copy_row(vvect_t<int>({1, 2, 3}), 1)
        );
    }},
    {"compute sigmoid zero weights", []() {
        // sigmoid(0) = 0.5
        Network<double> net(3, 3, 3, 3, activation_t::SIGMOID);
        vmatrix_t<double> out = net.compute(vmatrix_t<double>(1, 3));
        assert(out == vmatrix_t<double>(1, 3, {0.5, 0.5, 0.5}));
    }},
    {"compute batch", []() {
        Network<int> net(3, 3, 3, 3, activation_t::RELU);
        net.set_weight(0, vmatrix_t<int>::ID(3));
        net.set_weight(1, vmatrix_t<int>::ID(3));
        net.set_weight(2, vmatrix_t<int>::ID(3));
        vmatrix_t<int> in(2, 3, {
            1, 2, 3,
            4, 5, 6
        });
        assert(net.compute(in) == in);
    }}
});

int main() {
    ios::sync_with_stdio(false);
    cin.tie(NULL);
    cout << setprecision(15);

    tests.run();

    return 0;
}