#pragma once

#include <cassert>
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

    // matrix negative
    template <class type>
    inline vmatrix_t<type> operator -(const vmatrix_t<type>& m) {
        vmatrix_t<type> val(m.nr(), m.nc());
        #pragma omp simd
        for (int i = 0; i < m.raw_size(); i++) {val.raw_at(i) = -m.raw_at(i);}
        return val;
    }

    // matrix addition
    template <class type>
    inline vmatrix_t<type> operator +(const vmatrix_t<type>& m1, const vmatrix_t<type>& m2) {
        assert(m1.nr() == m2.nr() && m1.nc() == m2.nc());
        vmatrix_t<type> val(m1.nr(), m1.nc());
        #pragma omp simd
        for (int i = 0; i < m1.raw_size(); i++) {val.raw_at(i) = m1.raw_at(i)+m2.raw_at(i);}
        return val;
    }
    template <class type>
    inline vmatrix_t<type>& operator +=(vmatrix_t<type>& m1, const vmatrix_t<type>& m2) {
        assert(m1.nr() == m2.nr() && m1.nc() == m2.nc());
        #pragma omp simd
        for (int i = 0; i < m1.raw_size(); i++) {m1.raw_at(i) += m2.raw_at(i);}
        return m1;
    }
    template <class type>
    inline vmatrix_t<type> operator -(const vmatrix_t<type>& m1, const vmatrix_t<type>& m2) {
        assert(m1.nr() == m2.nr() && m1.nc() == m2.nc());
        vmatrix_t<type> val(m1.nr(), m1.nc());
        #pragma omp simd
        for (int i = 0; i < m1.raw_size(); i++) {val.raw_at(i) = m1.raw_at(i)-m2.raw_at(i);}
        return val;
    }
    template <class type>
    inline vmatrix_t<type>& operator -=(vmatrix_t<type>& m1, const vmatrix_t<type>& m2) {
        assert(m1.nr() == m2.nr() && m1.nc() == m2.nc());
        #pragma omp simd
        for (int i = 0; i < m1.raw_size(); i++) {m1.raw_at(i) -= m2.raw_at(i);}
        return m1;
    }

    // matrix scalar multiplication
    template <class type>
    inline vmatrix_t<type> operator *(const vmatrix_t<type>& m, type scale) {
        vmatrix_t<type> val(m.nr(), m.nc());
        #pragma omp simd
        for (int i = 0; i < m.raw_size(); i++) {val.raw_at(i) = m.raw_at(i)*scale;}
        return val;
    }
    template <class type>
    inline vmatrix_t<type>& operator *=(vmatrix_t<type>& m, type scale) {
        #pragma omp simd
        for (int i = 0; i < m.raw_size(); i++) {m.raw_at(i) *= scale;}
        return m;
    }
    template <class type>
    inline vmatrix_t<type> operator /(const vmatrix_t<type>& m, type scale) {
        vmatrix_t<type> val(m.nr(), m.nc());
        #pragma omp simd
        for (int i = 0; i < m.raw_size(); i++) {val.raw_at(i) = m.raw_at(i)/scale;}
        return val;
    }
    template <class type>
    inline vmatrix_t<type>& operator /=(vmatrix_t<type>& m, type scale) {
        #pragma omp simd
        for (int i = 0; i < m.raw_size(); i++) {m.raw_at(i) /= scale;}
        return m;
    }

    // vector matrix multiplication
    template <class type>
    inline vvect_t<type> operator *(const vvect_t<type>& v, const vmatrix_t<type>& m) {
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
    inline vvect_t<type>& operator *=(vvect_t<type>& v, const vmatrix_t<type>& m) {
        return v = v*m;
    }

    // matrix matrix multiplication
    template <class type>
    inline vmatrix_t<type> operator *(const vmatrix_t<type>& m1, const vmatrix_t<type>& m2) {
        assert(m1.nc() == m2.nr());
        vmatrix_t<type> val(m1.nr(), m2.nc());
        // iterating j on the innermost is better for cache locality
        for (int i = 0; i < m1.nr(); i++) {
            for (int k = 0; k < m1.nc(); k++) {
                #pragma omp simd
                for (int j = 0; j < m2.nc(); j++) {val.at(i, j) += m1.at(i, k)*m2.at(k, j);}
            }
        }
        return val;
    }
    template <class type>
    inline vmatrix_t<type>& operator *=(vmatrix_t<type>& m1, const vmatrix_t<type>& m2) {
        return m1 = m1*m2;
    }
}
