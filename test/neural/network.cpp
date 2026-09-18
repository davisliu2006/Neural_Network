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
        Network<int> net(3, 2, 4, 5, relu<int>);
        assert(net.in_size() == 3 && net.out_size() == 2);
        assert(net.n_layer() == 4 && net.layer_size() == 5);
        // first layer size
        assert(net.get_transform(0).nr() == 3 && net.get_transform(0).nc() == 5);
        assert(net.get_bias(0).size() == 5);
        // middle layer sizes
        for (int i = 1; i < net.n_layer()-1; i++) {
            assert(net.get_transform(i).nr() == 5 && net.get_transform(i).nc() == 5);
            assert(net.get_bias(i).size() == 5);
        }
        // last layer size
        assert(net.get_transform(3).nr() == 5 && net.get_transform(3).nc() == 2);
        assert(net.get_bias(3).size() == 2);
        // compute
        assert(net.compute(vvect_t<int>({1, 2, 3})) == vvect_t<int>(2));
    }},
    {"compute identity network", []() {
        Network<int> net(3, 3, 3, 3, relu<int>);
        net.set_transform(0, vmatrix_t<int>::ID(3));
        net.set_transform(1, vmatrix_t<int>::ID(3));
        net.set_transform(2, vmatrix_t<int>::ID(3));
        assert(net.compute(vvect_t<int>({1, 2, 3})) == vvect_t<int>({1, 2, 3}));
    }},
    {"compute sigmoid zero weights", []() {
        // sigmoid(0) = 0.5
        Network<double> net(3, 3, 3, 3, sigmoid<double>);
        vvect_t<double> out = net.compute(vvect_t<double>(3, 0));
        assert(out == vvect_t<double>(3, 0.5));
    }}
});

int main() {
    ios::sync_with_stdio(false);
    cin.tie(NULL);
    cout << setprecision(15);

    tests.run();

    return 0;
}
