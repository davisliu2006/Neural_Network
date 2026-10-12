#pragma once

#include "matrix.hpp"

namespace lin_alg {
    template <class type>
    struct [[deprecated("This is experimental")]] matrix_view {
        virtual int nr() const = 0;
        virtual int nc() const = 0;
        virtual type at(int i, int j) const = 0;

        vmatrix_t<type> materialize() const {
            vmatrix_t<type> val(nr(), nc());
            for (int i = 0; i < nr(); i++) {
                for (int j = 0; j < nc(); j++) {
                    val.at(i, j) = at(i, j);
                }
            }
            return val;
        }
    };

    template <class type>
    struct [[deprecated("This is experimental")]] transpose_view: matrix_view<type> {
        const vmatrix_t<type>& _orig;

        transpose_view(const vmatrix_t<type>& orig): _orig(orig) {}

        int nr() const override {return _orig.nc();}
        int nc() const override {return _orig.nr();}
        type at(int i, int j) const override {return _orig.at(j, i);}
    };

    template <class type>
    struct [[deprecated("This is experimental")]] copy_row_view: matrix_view<type> {
        const vvect_t<type>& _orig;
        int _nr;

        copy_row_view(const vvect_t<type>& orig, int nr): _orig(orig), _nr(nr) {}

        int nr() const override {return _nr;}
        int nc() const override {return _orig.size();}
        type at(int, int j) const override {return _orig[j];}
    };

    template <class type>
    struct [[deprecated("This is experimental")]] copy_col_view: matrix_view<type> {
        const vvect_t<type>& _orig;
        int _nc;

        copy_col_view(const vvect_t<type>& orig, int nc): _orig(orig), _nc(nc) {}

        int nr() const override {return _orig.size();}
        int nc() const override {return _nc;}
        type at(int i, int) const override {return _orig[i];}
    };

    // matrix addition
    template <class type>
    inline vmatrix_t<type>& operator +=(vmatrix_t<type>& m1, const matrix_view<type>& m2) {
        assert(m1.nr() == m2.nr() && m1.nc() == m2.nc());
        for (int i = 0; i < m1.nr(); i++) {
            #pragma omp simd
            for (int j = 0; j < m1.nc(); j++) {m1.at(i, j) += m2.at(i, j);}
        }
        return m1;
    }
    template <class type>
    inline vmatrix_t<type> operator +(const vmatrix_t<type>& m1, const matrix_view<type>& m2) {
        vmatrix_t<type> val = m1;
        val += m2;
        return val;
    }
    template <class type>
    inline vmatrix_t<type>& operator -=(vmatrix_t<type>& m1, const matrix_view<type>& m2) {
        assert(m1.nr() == m2.nr() && m1.nc() == m2.nc());
        for (int i = 0; i < m1.nr(); i++) {
            #pragma omp simd
            for (int j = 0; j < m1.nc(); j++) {m1.at(i, j) -= m2.at(i, j);}
        }
        return m1;
    }
    template <class type>
    inline vmatrix_t<type> operator -(const vmatrix_t<type>& m1, const matrix_view<type>& m2) {
        vmatrix_t<type> val = m1;
        val -= m2;
        return val;
    }

    // vector matrix multiplication
    template <class type>
    inline vvect_t<type> operator *(const vvect_t<type>& v, const matrix_view<type>& m) {
        assert(v.size() == m.nr());
        vvect_t<type> val(m.nc());
        // iterating j on the innermost is better for cache locality
        for (int i = 0; i < m.nr(); i++) {
            #pragma omp simd
            for (int j = 0; j < m.nc(); j++) {val[j] += v[i]*m.at(i, j);}
        }
        return val;
    }
    template <class type>
    inline vvect_t<type>& operator *=(vvect_t<type>& v, const matrix_view<type>& m) {
        return v = v*m;
    }

    // element-wise multiplication
    template <class type>
    inline vmatrix_t<type>& apply_elem_prod(vmatrix_t<type>& m1, const matrix_view<type>& m2) {
        assert(m1.nr() == m2.nr() && m1.nc() == m2.nc());
        for (int i = 0; i < m1.nr(); i++) {
            #pragma omp simd
            for (int j = 0; j < m1.nc(); j++) {m1.at(i, j) *= m2.at(i, j);}
        }
        return m1;
    }
    template <class type>
    inline vmatrix_t<type> elem_prod(const vmatrix_t<type>& m1, const matrix_view<type>& m2) {
        vmatrix_t<type> val = m1;
        apply_elem_prod(val, m2);
        return val;
    }
    template <class type>
    inline vmatrix_t<type> elem_prod(const matrix_view<type>& m1, const matrix_view<type>& m2) {
        vmatrix_t<type> val = m1.materialize();
        apply_elem_prod(val, m2);
        return val;
    }
}