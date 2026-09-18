#pragma once

#include <type_traits>

namespace neural {
    template <class type>
    using enable_if_float_t = std::enable_if_t<
        std::is_floating_point_v<type>, int
    >;
}