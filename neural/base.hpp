#pragma once

#include <type_traits>

namespace neural {
    template <class ret_t, class... arg_t>
    using function = ret_t (*)(arg_t...);

    template <class type>
    using enable_if_float_t = std::enable_if_t<
        std::is_floating_point_v<type>, int
    >;

    template <class type>
    inline type sq(type x) {return x*x;}
}