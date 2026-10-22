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
    inline void assign_add(
        vmatrix_t<type>& dst, const vmatrix_t<type>& m1, const matrix_view<View>& m2
    ) {
        assert(dst.nr() == m1.nr() && dst.nc() == m1.nc());
        assert(m1.nr() == m2.nr() && m1.nc() == m2.nc());
        for (int i = 0; i < dst.nr(); i++) {
            #pragma omp simd
            for (int j = 0; j < dst.nc(); j++) {dst.at(i, j) = m1.at(i, j) + m2.at(i, j);}
        }
    }
    template <class type, class View>
    inline vmatrix_t<type>& operator +=(vmatrix_t<type>& m1, const matrix_view<View>& m2) {
        assign_add(m1, m1, m2);
        return m1;
    }
    template <class type, class View>
    inline vmatrix_t<type> operator +(const vmatrix_t<type>& m1, const matrix_view<View>& m2) {
        vmatrix_t<type> val(m1.nr(), m1.nc());
        assign_add(val, m1, m2);
        return val;
    }
    template <class type, class View>
    inline void assign_sub(
        vmatrix_t<type>& dst, const vmatrix_t<type>& m1, const matrix_view<View>& m2
    ) {
        assert(dst.nr() == m1.nr() && dst.nc() == m1.nc());
        assert(m1.nr() == m2.nr() && m1.nc() == m2.nc());
        for (int i = 0; i < dst.nr(); i++) {
            #pragma omp simd
            for (int j = 0; j < dst.nc(); j++) {dst.at(i, j) = m1.at(i, j) - m2.at(i, j);}
        }
    }
    template <class type, class View>
    inline vmatrix_t<type>& operator -=(vmatrix_t<type>& m1, const matrix_view<View>& m2) {
        assign_sub(m1, m1, m2);
        return m1;
    }
    template <class type, class View>
    inline vmatrix_t<type> operator -(const vmatrix_t<type>& m1, const matrix_view<View>& m2) {
        vmatrix_t<type> val(m1.nr(), m1.nc());
        assign_sub(val, m1, m2);
        return val;
    }

    // vector matrix multiplication
    // special case for transpose
    template <class type>
    inline void assign_mul(
        vvect_t<type>& dst,
        const vvect_t<type>& v,
        const matrix_view<transpose_view<type>>& m
    ) {
        const vmatrix_t<type>& w = m._view._orig;
        assert(&dst != &v);
        assert(v.size() == w.nc());
        assert(dst.size() == w.nr());
        for (int j = 0; j < w.nr(); j++) {
            type sum = 0;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < w.nc(); k++) {sum += v[k]*w.at(j, k);}
            dst[j] = sum;
        }
    }
    template <class type>
    inline vvect_t<type> operator *(
        const vvect_t<type>& v, const matrix_view<transpose_view<type>>& m
    ) {
        vvect_t<type> val(m.nc());
        assign_mul(val, v, m);
        return val;
    }
    template <class type, class View>
    inline void assign_mul(
        vvect_t<type>& dst, const vvect_t<type>& v, const matrix_view<View>& m
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
    template <class type, class View>
    inline vvect_t<type> operator *(const vvect_t<type>& v, const matrix_view<View>& m) {
        vvect_t<type> val(m.nc());
        assign_mul(val, v, m);
        return val;
    }
    template <class type, class View>
    inline vvect_t<type>& operator *=(vvect_t<type>& v, const matrix_view<View>& m) {
        return v = v*m;
    }

    // matrix times transpose view
    template <class type>
    inline void assign_mul(
        vmatrix_t<type>& dst,
        const vmatrix_t<type>& a,
        const matrix_view<transpose_view<type>>& b
    ) {
        const vmatrix_t<type>& w = b._view._orig;
        assert(&dst != &a);
        assert(a.nc() == w.nc());
        assert(dst.nr() == a.nr() && dst.nc() == w.nr());
        #pragma omp parallel for schedule(static) if(a.nr() >= 8)
        for (int i = 0; i < a.nr(); i++) {
            for (int j = 0; j < w.nr(); j++) {
                type sum = 0;
                #pragma omp simd reduction(+:sum)
                for (int k = 0; k < w.nc(); k++) {sum += a.at(i, k)*w.at(j, k);}
                dst.at(i, j) = sum;
            }
        }
    }
    template <class type>
    inline vmatrix_t<type> operator *(
        const vmatrix_t<type>& a, const matrix_view<transpose_view<type>>& b
    ) {
        vmatrix_t<type> val(a.nr(), b.nc());
        assign_mul(val, a, b);
        return val;
    }

    // transpose view times matrix
    template <class type>
    inline void assign_mul(
        vmatrix_t<type>& dst,
        const matrix_view<transpose_view<type>>& a,
        const vmatrix_t<type>& b
    ) {
        const vmatrix_t<type>& a_orig = a._view._orig;
        assert(&dst != &a_orig);
        assert(a_orig.nr() == b.nr());
        assert(dst.nr() == a_orig.nc() && dst.nc() == b.nc());
        #pragma omp simd
        for (int i = 0; i < dst.raw_size(); i++) {dst.raw_at(i) = type(0);}
        #pragma omp parallel for schedule(static) if(a_orig.nc() >= 8)
        for (int i = 0; i < a_orig.nc(); i++) {
            for (int k = 0; k < b.nr(); k++) {
                #pragma omp simd
                for (int j = 0; j < b.nc(); j++) {
                    dst.at(i, j) += a_orig.at(k, i)*b.at(k, j);
                }
            }
        }
    }
    template <class type>
    inline vmatrix_t<type> operator *(
        const matrix_view<transpose_view<type>>& a, const vmatrix_t<type>& b
    ) {
        vmatrix_t<type> val(a.nr(), b.nc());
        assign_mul(val, a, b);
        return val;
    }

    // element-wise multiplication
    template <class type, class View>
    inline void assign_elem_prod(
        vmatrix_t<type>& dst, const vmatrix_t<type>& m1, const matrix_view<View>& m2
    ) {
        assert(dst.nr() == m1.nr() && dst.nc() == m1.nc());
        assert(m1.nr() == m2.nr() && m1.nc() == m2.nc());
        for (int i = 0; i < dst.nr(); i++) {
            #pragma omp simd
            for (int j = 0; j < dst.nc(); j++) {
                dst.at(i, j) = m1.at(i, j) * m2.at(i, j);
            }
        }
    }
    template <class type, class View>
    inline vmatrix_t<type>& apply_elem_prod(vmatrix_t<type>& m1, const matrix_view<View>& m2) {
        assign_elem_prod(m1, m1, m2);
        return m1;
    }
    template <class type, class View>
    inline vmatrix_t<type> elem_prod(const vmatrix_t<type>& m1, const matrix_view<View>& m2) {
        vmatrix_t<type> val(m1.nr(), m1.nc());
        assign_elem_prod(val, m1, m2);
        return val;
    }
}