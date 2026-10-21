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

    // matrix in-place assign
    template <class type>
    inline void assign(vmatrix_t<type>& dst, const vmatrix_t<type>& src) {
        assert(dst.nr() == src.nr() && dst.nc() == src.nc());
        for (int i = 0; i < dst.nr(); i++) {
            #pragma omp simd
            for (int j = 0; j < dst.nc(); j++) {dst.at(i, j) = src.at(i, j);}
        }
    }

    // matrix negative
    template <class type>
    inline vmatrix_t<type>& apply_neg(vmatrix_t<type>& m) {
        #pragma omp simd
        for (int i = 0; i < m.raw_size(); i++) {m.raw_at(i) = -m.raw_at(i);}
        return m;
    }
    template <class type>
    inline vmatrix_t<type> operator -(const vmatrix_t<type>& m) {
        vmatrix_t<type> val = m;
        apply_neg(val);
        return val;
    }

    // matrix addition
    template <class type>
    inline vmatrix_t<type>& operator +=(vmatrix_t<type>& m1, const vmatrix_t<type>& m2) {
        assert(m1.nr() == m2.nr() && m1.nc() == m2.nc());
        #pragma omp simd
        for (int i = 0; i < m1.raw_size(); i++) {m1.raw_at(i) += m2.raw_at(i);}
        return m1;
    }
    template <class type>
    inline vmatrix_t<type> operator +(const vmatrix_t<type>& m1, const vmatrix_t<type>& m2) {
        vmatrix_t<type> val = m1;
        val += m2;
        return val;
    }
    template <class type>
    inline vmatrix_t<type>& operator -=(vmatrix_t<type>& m1, const vmatrix_t<type>& m2) {
        assert(m1.nr() == m2.nr() && m1.nc() == m2.nc());
        #pragma omp simd
        for (int i = 0; i < m1.raw_size(); i++) {m1.raw_at(i) -= m2.raw_at(i);}
        return m1;
    }
    template <class type>
    inline vmatrix_t<type> operator -(const vmatrix_t<type>& m1, const vmatrix_t<type>& m2) {
        vmatrix_t<type> val = m1;
        val -= m2;
        return val;
    }

    // matrix scalar multiplication
    template <class type>
    inline vmatrix_t<type>& operator *=(vmatrix_t<type>& m, type scale) {
        #pragma omp simd
        for (int i = 0; i < m.raw_size(); i++) {m.raw_at(i) *= scale;}
        return m;
    }
    template <class type>
    inline vmatrix_t<type> operator *(const vmatrix_t<type>& m, type scale) {
        vmatrix_t<type> val = m;
        val *= scale;
        return val;
    }
    template <class type>
    inline vmatrix_t<type>& operator /=(vmatrix_t<type>& m, type scale) {
        #pragma omp simd
        for (int i = 0; i < m.raw_size(); i++) {m.raw_at(i) /= scale;}
        return m;
    }
    template <class type>
    inline vmatrix_t<type> operator /(const vmatrix_t<type>& m, type scale) {
        vmatrix_t<type> val = m;
        val /= scale;
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
    inline vvect_t<type> sum_rows(const vmatrix_t<type>& m) {
        vvect_t<type> v(m.nc());
        for (int i = 0; i < m.nr(); i++) {
            #pragma omp simd
            for (int j = 0; j < m.nc(); j++) {v[j] += m.at(i, j);}
        }
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
        assert(m1.nc() == m2.nr());
        assert(dst.nr() == m1.nr() && dst.nc() == m2.nc());
        #pragma omp simd
        for (int i = 0; i < dst.raw_size(); i++) {dst.raw_at(i) = type(0);}
        // iterating j on the innermost is better for cache locality
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
    inline vmatrix_t<type>& apply_elem_prod(vmatrix_t<type>& m1, const vmatrix_t<type>& m2) {
        assert(m1.nr() == m2.nr() && m1.nc() == m2.nc());
        #pragma omp simd
        for (int i = 0; i < m1.raw_size(); i++) {m1.raw_at(i) *= m2.raw_at(i);}
        return m1;
    }
    template <class type>
    inline vmatrix_t<type> elem_prod(const vmatrix_t<type>& m1, const vmatrix_t<type>& m2) {
        vmatrix_t<type> val = m1;
        apply_elem_prod(val, m2);
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