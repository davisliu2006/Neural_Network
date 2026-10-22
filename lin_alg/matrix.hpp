#pragma once

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "vector.hpp"

namespace lin_alg {
    template <class type = double>
    struct vmatrix_t {
        private:
        int _nr, _nc;
        std::vector<type> arr;

        public:
        vmatrix_t(): _nr(0), _nc(0), arr() {}
        vmatrix_t(int nr, int nc): _nr(nr), _nc(nc), arr(nr*nc) {}
        vmatrix_t(int nr, int nc, const std::vector<type>& arr):
        _nr(nr), _nc(nc) {
            assert(arr.size() == nr*nc);
            this->arr = arr;
        }

        int nr() const {return _nr;}
        int nc() const {return _nc;}
        type& at(int i, int j) {return arr[i*nc() + j];}
        type at(int i, int j) const {return arr[i*nc() + j];}

        int raw_size() const {return nr()*nc();}
        type& raw_at(int i) {return arr[i];}
        type raw_at(int i) const {return arr[i];}

        /**
         * Grows element storage if needed. Does not change logical shape.
         */
        void ensure_capacity(int nr, int nc) {
            assert(nr >= 0 && nc >= 0);
            size_t need = size_t(nr)*size_t(nc);
            if (need > arr.size()) {arr.resize(need);}
        }
        /**
         * Sets logical shape. Storage grows if needed.
         */
        void set_shape(int nr, int nc) {
            ensure_capacity(nr, nc);
            _nr = nr;
            _nc = nc;
        }

        double vect_mag2() const {
            double val = 0;
            #pragma omp simd reduction(+:val)
            for (int i = 0; i < raw_size(); i++) {val += arr[i]*arr[i];}
            return val;
        }
        double vect_mag() const {
            return std::sqrt(vect_mag2());
        }

        vmatrix_t<type> transpose() const {
            vmatrix_t<type> val(nc(), nr());
            for (int i = 0; i < nr(); i++) {
                for (int j = 0; j < nc(); j++) {val.at(j, i) = at(i, j);}
            }
            return val;
        }

        static vmatrix_t<type> ID(int n) {
            vmatrix_t<type> val(n, n);
            for (int i = 0; i < n; i++) {val.at(i, i) = 1;}
            return val;
        }
    };

    // matrix print
    template <class type>
    inline std::ostream& operator <<(std::ostream& out, const vmatrix_t<type>& m) {
        assert(m.nr() > 0 && m.nc() > 0);
        out << "[\n";
        for (int i = 0; i < m.nr(); i++) {
            out << "\t[";
            for (int j = 0; j < m.nc()-1; j++) {out << m.at(i, j) << ", ";}
            out << m.at(i, m.nc()-1) << "]\n";
        }
        out << ']';
        return out;
    }


    // matrix equals
    template <class type>
    inline bool operator ==(const vmatrix_t<type>& m1, const vmatrix_t<type>& m2) {
        assert(m1.nr() == m2.nr() && m1.nc() == m2.nc());
        for (int i = 0; i < m1.raw_size(); i++) {if (m1.raw_at(i) != m2.raw_at(i)) {return false;}}
        return true;
    }

    // matrix assign
    template <class type>
    inline void assign(vmatrix_t<type>& dst, const vmatrix_t<type>& src) {
        assert(dst.nr() == src.nr() && dst.nc() == src.nc());
        for (int i = 0; i < dst.nr(); i++) {
            #pragma omp simd
            for (int j = 0; j < dst.nc(); j++) {dst.at(i, j) = src.at(i, j);}
        }
    }
    template <class type>
    inline void assign_elems(vmatrix_t<type>& dst, type val) {
        #pragma omp simd
        for (int i = 0; i < dst.raw_size(); i++) {dst.raw_at(i) = val;}
    }

    // matrix negative
    template <class type>
    inline void assign_neg(vmatrix_t<type>& dst, const vmatrix_t<type>& m) {
        assert(dst.nr() == m.nr() && dst.nc() == m.nc());
        #pragma omp simd
        for (int i = 0; i < dst.raw_size(); i++) {dst.raw_at(i) = -m.raw_at(i);}
    }
    template <class type>
    inline vmatrix_t<type>& apply_neg(vmatrix_t<type>& m) {
        assign_neg(m, m);
        return m;
    }
    template <class type>
    inline vmatrix_t<type> operator -(const vmatrix_t<type>& m) {
        vmatrix_t<type> val(m.nr(), m.nc());
        assign_neg(val, m);
        return val;
    }

    // matrix addition
    template <class type>
    inline void assign_add(
        vmatrix_t<type>& dst, const vmatrix_t<type>& m1, const vmatrix_t<type>& m2
    ) {
        assert(dst.nr() == m1.nr() && dst.nc() == m1.nc());
        assert(m1.nr() == m2.nr() && m1.nc() == m2.nc());
        #pragma omp simd
        for (int i = 0; i < dst.raw_size(); i++) {
            dst.raw_at(i) = m1.raw_at(i) + m2.raw_at(i);
        }
    }
    template <class type>
    inline vmatrix_t<type>& operator +=(vmatrix_t<type>& m1, const vmatrix_t<type>& m2) {
        assign_add(m1, m1, m2);
        return m1;
    }
    template <class type>
    inline vmatrix_t<type> operator +(const vmatrix_t<type>& m1, const vmatrix_t<type>& m2) {
        vmatrix_t<type> val(m1.nr(), m1.nc());
        assign_add(val, m1, m2);
        return val;
    }
    template <class type>
    inline void assign_sub(
        vmatrix_t<type>& dst, const vmatrix_t<type>& m1, const vmatrix_t<type>& m2
    ) {
        assert(dst.nr() == m1.nr() && dst.nc() == m1.nc());
        assert(m1.nr() == m2.nr() && m1.nc() == m2.nc());
        #pragma omp simd
        for (int i = 0; i < dst.raw_size(); i++) {
            dst.raw_at(i) = m1.raw_at(i) - m2.raw_at(i);
        }
    }
    template <class type>
    inline vmatrix_t<type>& operator -=(vmatrix_t<type>& m1, const vmatrix_t<type>& m2) {
        assign_sub(m1, m1, m2);
        return m1;
    }
    template <class type>
    inline vmatrix_t<type> operator -(const vmatrix_t<type>& m1, const vmatrix_t<type>& m2) {
        vmatrix_t<type> val(m1.nr(), m1.nc());
        assign_sub(val, m1, m2);
        return val;
    }

    // matrix scalar multiplication
    template <class type>
    inline void assign_mul(vmatrix_t<type>& dst, const vmatrix_t<type>& m, type scale) {
        assert(dst.nr() == m.nr() && dst.nc() == m.nc());
        #pragma omp simd
        for (int i = 0; i < dst.raw_size(); i++) {dst.raw_at(i) = m.raw_at(i) * scale;}
    }
    template <class type>
    inline vmatrix_t<type>& operator *=(vmatrix_t<type>& m, type scale) {
        assign_mul(m, m, scale);
        return m;
    }
    template <class type>
    inline vmatrix_t<type> operator *(const vmatrix_t<type>& m, type scale) {
        vmatrix_t<type> val(m.nr(), m.nc());
        assign_mul(val, m, scale);
        return val;
    }
    template <class type>
    inline void assign_div(vmatrix_t<type>& dst, const vmatrix_t<type>& m, type scale) {
        assert(dst.nr() == m.nr() && dst.nc() == m.nc());
        #pragma omp simd
        for (int i = 0; i < dst.raw_size(); i++) {dst.raw_at(i) = m.raw_at(i) / scale;}
    }
    template <class type>
    inline vmatrix_t<type>& operator /=(vmatrix_t<type>& m, type scale) {
        assign_div(m, m, scale);
        return m;
    }
    template <class type>
    inline vmatrix_t<type> operator /(const vmatrix_t<type>& m, type scale) {
        vmatrix_t<type> val(m.nr(), m.nc());
        assign_div(val, m, scale);
        return val;
    }

    // add vector to each row
    template <class type>
    inline vmatrix_t<type>& apply_add_row(vmatrix_t<type>& m, const vvect_t<type>& v) {
        assert(m.nc() == v.size());
        for (int i = 0; i < m.nr(); i++) {
            #pragma omp simd
            for (int j = 0; j < m.nc(); j++) {m.at(i, j) += v[j];}
        }
        return m;
    }

    // sum over rows
    template <class type>
    inline void assign_sum_rows(vvect_t<type>& dst, const vmatrix_t<type>& m) {
        assert(dst.size() == m.nc());
        for (int j = 0; j < dst.size(); j++) {dst[j] = type(0);}
        for (int i = 0; i < m.nr(); i++) {
            #pragma omp simd
            for (int j = 0; j < m.nc(); j++) {dst[j] += m.at(i, j);}
        }
    }
    template <class type>
    inline vvect_t<type> sum_rows(const vmatrix_t<type>& m) {
        vvect_t<type> v(m.nc());
        assign_sum_rows(v, m);
        return v;
    }

    // extract a row as a vector
    template <class type>
    inline vvect_t<type> get_row(const vmatrix_t<type>& m, int i) {
        assert(0 <= i && i < m.nr());
        vvect_t<type> v(m.nc());
        for (int j = 0; j < m.nc(); j++) {v[j] = m.at(i, j);}
        return v;
    }

    // vector matrix multiplication
    template <class type>
    inline void assign_mul(
        vvect_t<type>& dst, const vvect_t<type>& v, const vmatrix_t<type>& m
    ) {
        assert(&dst != &v);
        assert(v.size() == m.nr());
        assert(dst.size() == m.nc());
        for (int j = 0; j < dst.size(); j++) {dst[j] = type(0);}
        // iterating j on the innermost is better for cache locality
        for (int i = 0; i < m.nr(); i++) {
            #pragma omp simd
            for (int j = 0; j < m.nc(); j++) {dst[j] += v[i]*m.at(i, j);}
        }
    }
    template <class type>
    inline vvect_t<type> operator *(const vvect_t<type>& v, const vmatrix_t<type>& m) {
        vvect_t<type> val(m.nc());
        assign_mul(val, v, m);
        return val;
    }
    template <class type>
    inline vvect_t<type>& operator *=(vvect_t<type>& v, const vmatrix_t<type>& m) {
        return v = v*m;
    }

    // matrix matrix multiplication
    template <class type>
    inline void assign_mul(
        vmatrix_t<type>& dst, const vmatrix_t<type>& m1, const vmatrix_t<type>& m2
    ) {
        assert(&dst != &m1);
        assert(m1.nc() == m2.nr());
        assert(dst.nr() == m1.nr() && dst.nc() == m2.nc());
        #pragma omp simd
        for (int i = 0; i < dst.raw_size(); i++) {dst.raw_at(i) = type(0);}
        // parallel over i
        // iterating j on the innermost is better for cache locality
        #pragma omp parallel for schedule(static) if(m1.nr() >= 8)
        for (int i = 0; i < m1.nr(); i++) {
            for (int k = 0; k < m1.nc(); k++) {
                #pragma omp simd
                for (int j = 0; j < m2.nc(); j++) {
                    dst.at(i, j) += m1.at(i, k)*m2.at(k, j);
                }
            }
        }
    }
    template <class type>
    inline vmatrix_t<type> operator *(const vmatrix_t<type>& m1, const vmatrix_t<type>& m2) {
        vmatrix_t<type> val(m1.nr(), m2.nc());
        assign_mul(val, m1, m2);
        return val;
    }
    template <class type>
    inline vmatrix_t<type>& operator *=(vmatrix_t<type>& m1, const vmatrix_t<type>& m2) {
        return m1 = m1*m2;
    }

    // element-wise multiplication
    template <class type>
    inline void assign_elem_prod(
        vmatrix_t<type>& dst, const vmatrix_t<type>& m1, const vmatrix_t<type>& m2
    ) {
        assert(dst.nr() == m1.nr() && dst.nc() == m1.nc());
        assert(m1.nr() == m2.nr() && m1.nc() == m2.nc());
        #pragma omp simd
        for (int i = 0; i < dst.raw_size(); i++) {
            dst.raw_at(i) = m1.raw_at(i) * m2.raw_at(i);
        }
    }
    template <class type>
    inline vmatrix_t<type>& apply_elem_prod(vmatrix_t<type>& m1, const vmatrix_t<type>& m2) {
        assign_elem_prod(m1, m1, m2);
        return m1;
    }
    template <class type>
    inline vmatrix_t<type> elem_prod(const vmatrix_t<type>& m1, const vmatrix_t<type>& m2) {
        vmatrix_t<type> val(m1.nr(), m1.nc());
        assign_elem_prod(val, m1, m2);
        return val;
    }

    // vector copy row
    template <class type>
    inline vmatrix_t<type> copy_row(const vvect_t<type>& v, int nr) {
        vmatrix_t<type> val(nr, v.size());
        for (int i = 0; i < nr; i++) {
            for (int j = 0; j < v.size(); j++) {val.at(i, j) = v[j];}
        }
        return val;
    }
    
    // vector copy column
    template <class type>
    inline vmatrix_t<type> copy_col(const vvect_t<type>& v, int nc) {
        vmatrix_t<type> val(v.size(), nc);
        for (int i = 0; i < v.size(); i++) {
            for (int j = 0; j < nc; j++) {val.at(i, j) = v[i];}
        }
        return val;
    }
}