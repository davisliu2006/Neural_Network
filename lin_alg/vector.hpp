#pragma once

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

namespace lin_alg {
    template <class type = double>
    struct vvect_t {
        using iterator = typename std::vector<type>::iterator;
        using const_iterator = typename std::vector<type>::const_iterator;

        private:
        std::vector<type> arr;

        public:
        vvect_t(int n): arr(n) {}
        vvect_t(const std::vector<type>& arr): arr(arr) {}
        vvect_t(int n, type val): arr(n, val) {}

        int size() const {return arr.size();}
        type& operator [](int i) {return arr[i];}
        type operator [](int i) const {return arr[i];}

        double mag2() const {
            double val = 0;
            #pragma omp simd reduction(+:val)
            for (int i = 0; i < size(); i++) {val += arr[i]*arr[i];}
            return val;
        }
        double mag() const {
            return std::sqrt(mag2());
        }

        iterator begin() {return arr.begin();}
        iterator end() {return arr.end();}
        const_iterator begin() const {return arr.begin();}
        const_iterator end() const {return arr.end();}
    };

    // vector print
    template <class type>
    inline std::ostream& operator <<(std::ostream& out, const vvect_t<type>& v) {
        assert(v.size() > 0);
        out << '[';
        for (int i = 0; i < v.size()-1; i++) {out << v[i] << ", ";}
        out << v[v.size()-1] << ']';
        return out;
    }

    // vector equals
    template <class type>
    inline bool operator ==(const vvect_t<type>& v1, const vvect_t<type>& v2) {
        assert(v1.size() == v2.size());
        for (int i = 0; i < v1.size(); i++) {if (v1[i] != v2[i]) {return false;}}
        return true;
    }

    // vector negative
    template <class type>
    inline vvect_t<type> operator -(const vvect_t<type>& v) {
        vvect_t<type> val(v.size());
        #pragma omp simd
        for (int i = 0; i < v.size(); i++) {val[i] = -v[i];}
        return val;
    }

    // vector addition
    template <class type>
    inline vvect_t<type> operator +(const vvect_t<type>& v1, const vvect_t<type>& v2) {
        assert(v1.size() == v2.size());
        vvect_t<type> val(v1.size());
        #pragma omp simd
        for (int i = 0; i < v1.size(); i++) {val[i] = v1[i]+v2[i];}
        return val;
    }
    template <class type>
    inline vvect_t<type>& operator +=(vvect_t<type>& v1, const vvect_t<type>& v2) {
        assert(v1.size() == v2.size());
        #pragma omp simd
        for (int i = 0; i < v1.size(); i++) {v1[i] += v2[i];}
        return v1;
    }
    template <class type>
    inline vvect_t<type> operator -(const vvect_t<type>& v1, const vvect_t<type>& v2) {
        assert(v1.size() == v2.size());
        vvect_t<type> val(v1.size());
        #pragma omp simd
        for (int i = 0; i < v1.size(); i++) {val[i] = v1[i]-v2[i];}
        return val;
    }
    template <class type>
    inline vvect_t<type>& operator -=(vvect_t<type>& v1, const vvect_t<type>& v2) {
        assert(v1.size() == v2.size());
        #pragma omp simd
        for (int i = 0; i < v1.size(); i++) {v1[i] -= v2[i];}
        return v1;
    }

    // vector scalar multiplication
    template <class type>
    inline vvect_t<type> operator *(const vvect_t<type>& v, type scale) {
        vvect_t<type> val(v.size());
        #pragma omp simd
        for (int i = 0; i < v.size(); i++) {val[i] = v[i]*scale;}
        return val;
    }
    template <class type>
    inline vvect_t<type>& operator *=(vvect_t<type>& v, type scale) {
        #pragma omp simd
        for (int i = 0; i < v.size(); i++) {v[i] *= scale;}
        return v;
    }
    template <class type>
    inline vvect_t<type> operator /(const vvect_t<type>& v, type scale) {
        vvect_t<type> val(v.size());
        #pragma omp simd
        for (int i = 0; i < v.size(); i++) {val[i] = v[i]/scale;}
        return val;
    }
    template <class type>
    inline vvect_t<type>& operator /=(vvect_t<type>& v, type scale) {
        #pragma omp simd
        for (int i = 0; i < v.size(); i++) {v[i] /= scale;}
        return v;
    }

    // dot product
    template <class type>
    inline type dot_prod(const vvect_t<type>& v1, const vvect_t<type>& v2) {
        assert(v1.size() == v2.size());
        type val = 0;
        #pragma omp simd reduction(+:val)
        for (int i = 0; i < v1.size(); i++) {val += v1[i]*v2[i];}
        return val;
    }

    // element-wise multiplication
    template <class type>
    inline vvect_t<type> elem_prod(const vvect_t<type>& v1, const vvect_t<type>& v2) {
        assert(v1.size() == v2.size());
        vvect_t<type> val(v1.size());
        #pragma omp simd
        for (int i = 0; i < v1.size(); i++) {val[i] = v1[i]*v2[i];}
        return val;
    }
}