#pragma once

#include <chrono>
#include <type_traits>

namespace neural {
    // TYPES

    template <class ret_t, class... arg_t>
    using function = ret_t (*)(arg_t...);

    template <class type>
    using enable_if_float_t = std::enable_if_t<
        std::is_floating_point_v<type>, int
    >;

    // MATH

    template <class type>
    inline type sq(type x) {return x*x;}

    // TIME

    inline double time() {
        using namespace std::chrono;
        auto t = steady_clock::now();
        return duration_cast<microseconds>(t.time_since_epoch()).count()*1e-6;
    }
}