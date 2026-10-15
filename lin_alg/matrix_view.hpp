#pragma once

#include "matrix.hpp"

namespace lin_alg {
    template <class view_type>
    struct matrix_view {
        using value_type = typename view_type::value_type;
        view_type _view;

        matrix_view(const view_type& view): _view(view) {}

        int nr() const {return _view.nr();}
        int nc() const {return _view.nc();}
        value_type at(int i, int j) const {return _view.at(i, j);}

        vmatrix_t<value_type> materialize() const {
            vmatrix_t<value_type> val(nr(), nc());
            for (int i = 0; i < nr(); i++) {
                for (int j = 0; j < nc(); j++) {val.at(i, j) = at(i, j);}
            }
            return val;
        }
    };

    template <class type>
    struct transpose_view {
        using value_type = type;
        const vmatrix_t<type>& _orig;

        int nr() const {return _orig.nc();}
        int nc() const {return _orig.nr();}
        type at(int i, int j) const {return _orig.at(j, i);}
    };
    template <class type>
    struct copy_row_view {
        using value_type = type;
        const vvect_t<type>& _orig;
        int _nr;

        int nr() const {return _nr;}
        int nc() const {return _orig.size();}
        type at(int, int j) const {return _orig[j];}
    };
    template <class type>
    struct copy_col_view {
        using value_type = type;
        const vvect_t<type>& _orig;
        int _nc;

        int nr() const {return _orig.size();}
        int nc() const {return _nc;}
        type at(int i, int) const {return _orig[i];}
    };

    template <class A, class B>
    struct elem_prod_view {
        using value_type = typename A::value_type;
        matrix_view<A> _a;
        matrix_view<B> _b;

        int nr() const {return _a.nr();}
        int nc() const {return _a.nc();}
        value_type at(int i, int j) const {return _a.at(i, j)*_b.at(i, j);}
    };

    template <class type>
    inline matrix_view<transpose_view<type>> as_transpose(const vmatrix_t<type>& m) {
        return matrix_view<transpose_view<type>>(transpose_view<type>{m});
    }
    template <class type>
    inline matrix_view<copy_row_view<type>> as_copy_row(const vvect_t<type>& v, int nr) {
        return matrix_view<copy_row_view<type>>(copy_row_view<type>{v, nr});
    }
    template <class type>
    inline matrix_view<copy_col_view<type>> as_copy_col(const vvect_t<type>& v, int nc) {
        return matrix_view<copy_col_view<type>>(copy_col_view<type>{v, nc});
    }

    template <class A, class B>
    inline matrix_view<elem_prod_view<A, B>> as_elem_prod(
        const matrix_view<A>& a, const matrix_view<B>& b
    ) {
        assert(a.nr() == b.nr() && a.nc() == b.nc());
        return matrix_view<elem_prod_view<A, B>>(elem_prod_view<A, B>{a, b});
    }

    template <class type, class View>
    inline void assign(vmatrix_t<type>& dst, const matrix_view<View>& src) {
        assert(dst.nr() == src.nr() && dst.nc() == src.nc());
        for (int i = 0; i < dst.nr(); i++) {
            #pragma omp simd
            for (int j = 0; j < dst.nc(); j++) {dst.at(i, j) = src.at(i, j);}
        }
    }

    // matrix addition
    template <class type, class View>
    inline vmatrix_t<type>& operator +=(vmatrix_t<type>& m1, const matrix_view<View>& m2) {
        assert(m1.nr() == m2.nr() && m1.nc() == m2.nc());
        for (int i = 0; i < m1.nr(); i++) {
            #pragma omp simd
            for (int j = 0; j < m1.nc(); j++) {m1.at(i, j) += m2.at(i, j);}
        }
        return m1;
    }
    template <class type, class View>
    inline vmatrix_t<type> operator +(const vmatrix_t<type>& m1, const matrix_view<View>& m2) {
        vmatrix_t<type> val = m1;
        val += m2;
        return val;
    }
    template <class type, class View>
    inline vmatrix_t<type>& operator -=(vmatrix_t<type>& m1, const matrix_view<View>& m2) {
        assert(m1.nr() == m2.nr() && m1.nc() == m2.nc());
        for (int i = 0; i < m1.nr(); i++) {
            #pragma omp simd
            for (int j = 0; j < m1.nc(); j++) {m1.at(i, j) -= m2.at(i, j);}
        }
        return m1;
    }
    template <class type, class View>
    inline vmatrix_t<type> operator -(const vmatrix_t<type>& m1, const matrix_view<View>& m2) {
        vmatrix_t<type> val = m1;
        val -= m2;
        return val;
    }

    // vector matrix multiplication
    // special case for transpose
    template <class type>
    inline vvect_t<type> operator *(
        const vvect_t<type>& v, const matrix_view<transpose_view<type>>& m
    ) {
        const vmatrix_t<type>& w = m._view._orig;
        assert(v.size() == w.nc());
        vvect_t<type> val(w.nr());
        for (int j = 0; j < w.nr(); j++) {
            type sum = 0;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < w.nc(); k++) {sum += v[k]*w.at(j, k);}
            val[j] = sum;
        }
        return val;
    }
    template <class type, class View>
    inline vvect_t<type> operator *(const vvect_t<type>& v, const matrix_view<View>& m) {
        assert(v.size() == m.nr());
        vvect_t<type> val(m.nc());
        // iterating j on the innermost is better for cache locality
        for (int i = 0; i < m.nr(); i++) {
            #pragma omp simd
            for (int j = 0; j < m.nc(); j++) {val[j] += v[i]*m.at(i, j);}
        }
        return val;
    }
    template <class type, class View>
    inline vvect_t<type>& operator *=(vvect_t<type>& v, const matrix_view<View>& m) {
        return v = v*m;
    }

    // element-wise multiplication
    template <class type, class View>
    inline vmatrix_t<type>& apply_elem_prod(vmatrix_t<type>& m1, const matrix_view<View>& m2) {
        assert(m1.nr() == m2.nr() && m1.nc() == m2.nc());
        for (int i = 0; i < m1.nr(); i++) {
            #pragma omp simd
            for (int j = 0; j < m1.nc(); j++) {m1.at(i, j) *= m2.at(i, j);}
        }
        return m1;
    }
    template <class type, class View>
    inline vmatrix_t<type> elem_prod(const vmatrix_t<type>& m1, const matrix_view<View>& m2) {
        vmatrix_t<type> val = m1;
        apply_elem_prod(val, m2);
        return val;
    }
}