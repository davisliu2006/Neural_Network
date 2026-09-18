#pragma once

#include <string>

#include "matrix.hpp"
#include "vector.hpp"

namespace lin_alg {
    template <class type>
    struct IOHelper {
        static constexpr int TSIZE = sizeof(type);

        static std::string v_to_raw(const vvect_t<type>& v) {
            int data_size = TSIZE*v.size();
            std::string val(data_size, 0);
            memcpy(&val[0], &v[0], data_size);
            return val;
        }
        static vvect_t<type> raw_to_v(const std::string& str) {
            int data_size = str.size();
            assert(data_size % TSIZE == 0);
            vvect_t<type> val(data_size/TSIZE);
            memcpy(&val[0], &str[0], data_size);
            return val;
        }

        static std::string mat_to_raw(const vmatrix_t<type>& m) {
            int data_size = TSIZE*m.raw_size();
            std::string val(data_size, 0);
            memcpy(&val[0], &m.raw_at(0), data_size);
            return val;
        }
        static vmatrix_t<type> raw_to_mat(int nr, int nc, const std::string& str) {
            int data_size = str.size();
            assert(data_size == TSIZE*nr*nc);
            vmatrix_t<type> val(nr, nc);
            memcpy(&val.raw_at(0), &str[0], data_size);
            return val;
        }
    };
}