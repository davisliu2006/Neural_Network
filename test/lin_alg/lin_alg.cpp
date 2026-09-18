#include <cassert>
#include <iomanip>
#include <iostream>

#include "../test.hpp"
#include "../../lin_alg/vector.hpp"
#include "../../lin_alg/matrix.hpp"

using namespace std;
using namespace lin_alg;

const vvect_t<int> V1({3, -1, 4});
const vvect_t<int> V2({-1, 5, -9});
const vmatrix_t<int> M1(3, 3, {
    3, -1, 4,
    -1, 5, -9,
    2, -6, 5
});
const vmatrix_t<int> M2(3, 3, {
    -3, 5, -8,
    9, -7, 9,
    -3, 2, -3
});
const vmatrix_t<int> M3(2, 3, {
    1, 2, 3,
    4, 5, 6
});
const vmatrix_t<int> M4(3, 2, {
    0, 1,
    -1, 2,
    -2, 3
});

TestBatch tests({
    {"vector basic", []() {
        vvect_t<int> v1 = V1;
        vvect_t<int> v = v1*2 + V2;
        assert(v == vvect_t<int>({5, 3, -1}));
        assert(((v1 *= 2) += V2) == v);
        v = (v1-V2)/2;
        assert(v == V1);
        assert(((v1 -= V2) /= 2) == v);
        assert(-v == vvect_t<int>({-3, 1, -4}));
    }},
    {"matrix basic", []() {
        vmatrix_t<int> m1 = M1;
        vmatrix_t<int> m = (m1+M2)*2;
        assert(m == vmatrix_t<int>(3, 3, {
            0, 8, -8,
            16, -4, 0,
            -2, -8, 4
        }));
        assert(((m1 += M2) *= 2) == m);
        m = m1/2 - M2;
        assert(m == M1);
        assert(((m1 /= 2) -= M2) == M1);
        assert(-m == vmatrix_t<int>(3, 3, {
            -3, 1, -4,
            1, -5, 9,
            -2, 6, -5
        }));
    }},
    {"matrix multiplication", []() {
        vmatrix_t<int> m1 = M1;
        vmatrix_t<int> m = m1*M2;
        assert(m == vmatrix_t<int>(3, 3, {
            -30, 30, -45,
            75, -58, 80,
            -75, 62, -85
        }));
        assert((m1 *= M2) == m);
        vvect_t<int> v1 = V1;
        vvect_t<int> v = v1*M2;
        assert(v == vvect_t<int>({-30, 30, -45}));
        assert((v1 *= M2) == v);
    }},
    {"matrix multiplication non-square", []() {
        vmatrix_t<int> m = M3*M4;
        assert(m == vmatrix_t<int>(2, 2, {
            -8, 14,
            -17, 32
        }));
        vvect_t<int> v = V1*M4;
        assert(v == vvect_t<int>(vector<int>{-7, 13}));
    }},
    {"matrix indexing", []() {
        vmatrix_t<int> m = M3;
        assert(m.nr() == 2 && m.nc() == 3);
        assert(m.at(0, 0) == 1 && m.at(1, 1) == 5);
        assert(m.raw_at(0) == 1 && m.raw_at(4) == 5);
    }}
});

int main() {
    ios::sync_with_stdio(false);
    cin.tie(NULL);
    cout << setprecision(15);

    tests.run();

    return 0;
}