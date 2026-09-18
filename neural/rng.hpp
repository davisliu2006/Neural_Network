#pragma once

#include <random>

#include "../lin_alg/matrix.hpp"
#include "../lin_alg/vector.hpp"
#include "base.hpp"

namespace neural {
    using namespace lin_alg;

    struct RNG {
        std::mt19937 _gen32;
        std::mt19937_64 _gen64;

        RNG(uint32_t seed = 0): _gen32(seed), _gen64(seed) {}
        void seed(uint32_t seed) {_gen32.seed(seed); _gen64.seed(seed);}
        void seed(uint64_t seed) {_gen64.seed(seed);}
        uint32_t rand32() {return _gen32();}
        uint64_t rand64() {return _gen64();}
    };

    inline RNG rng;

    template <class type, enable_if_float_t<type> = 0>
    void init_random(vmatrix_t<type>& m) {
        std::uniform_real_distribution<type> dist(-0.5, 0.5);
        for (int i = 0; i < m.raw_size(); i++) {
            m.raw_at(i) = dist(rng._gen32);
        }
    }
}