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
        vvect_t(): arr() {}
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

    // vector assign
    template <class type>
    inline void assign(vvect_t<type>& dst, const vvect_t<type>& src) {
        assert(dst.size() == src.size());
        #pragma omp simd
        for (int i = 0; i < dst.size(); i++) {dst[i] = src[i];}
    }

    // vector negative
    template <class type>
    inline void assign_neg(vvect_t<type>& dst, const vvect_t<type>& v) {
        assert(dst.size() == v.size());
        #pragma omp simd
        for (int i = 0; i < dst.size(); i++) {dst[i] = -v[i];}
    }
    template <class type>
    inline vvect_t<type>& apply_neg(vvect_t<type>& v) {
        assign_neg(v, v);
        return v;
    }
    template <class type>
    inline vvect_t<type> operator -(const vvect_t<type>& v) {
        vvect_t<type> val(v.size());
        assign_neg(val, v);
        return val;
    }

    // vector addition
    template <class type>
    inline void assign_add(
        vvect_t<type>& dst, const vvect_t<type>& v1, const vvect_t<type>& v2
    ) {
        assert(dst.size() == v1.size() && v1.size() == v2.size());
        #pragma omp simd
        for (int i = 0; i < dst.size(); i++) {dst[i] = v1[i] + v2[i];}
    }
    template <class type>
    inline vvect_t<type>& operator +=(vvect_t<type>& v1, const vvect_t<type>& v2) {
        assign_add(v1, v1, v2);
        return v1;
    }
    template <class type>
    inline vvect_t<type> operator +(const vvect_t<type>& v1, const vvect_t<type>& v2) {
        vvect_t<type> val(v1.size());
        assign_add(val, v1, v2);
        return val;
    }
    template <class type>
    inline void assign_sub(
        vvect_t<type>& dst, const vvect_t<type>& v1, const vvect_t<type>& v2
    ) {
        assert(dst.size() == v1.size() && v1.size() == v2.size());
        #pragma omp simd
        for (int i = 0; i < dst.size(); i++) {dst[i] = v1[i] - v2[i];}
    }
    template <class type>
    inline vvect_t<type>& operator -=(vvect_t<type>& v1, const vvect_t<type>& v2) {
        assign_sub(v1, v1, v2);
        return v1;
    }
    template <class type>
    inline vvect_t<type> operator -(const vvect_t<type>& v1, const vvect_t<type>& v2) {
        vvect_t<type> val(v1.size());
        assign_sub(val, v1, v2);
        return val;
    }

    // vector scalar multiplication
    template <class type>
    inline void assign_mul(vvect_t<type>& dst, const vvect_t<type>& v, type scale) {
        assert(dst.size() == v.size());
        #pragma omp simd
        for (int i = 0; i < dst.size(); i++) {dst[i] = v[i] * scale;}
    }
    template <class type>
    inline vvect_t<type>& operator *=(vvect_t<type>& v, type scale) {
        assign_mul(v, v, scale);
        return v;
    }
    template <class type>
    inline vvect_t<type> operator *(const vvect_t<type>& v, type scale) {
        vvect_t<type> val(v.size());
        assign_mul(val, v, scale);
        return val;
    }
    template <class type>
    inline void assign_div(vvect_t<type>& dst, const vvect_t<type>& v, type scale) {
        assert(dst.size() == v.size());
        #pragma omp simd
        for (int i = 0; i < dst.size(); i++) {dst[i] = v[i] / scale;}
    }
    template <class type>
    inline vvect_t<type>& operator /=(vvect_t<type>& v, type scale) {
        assign_div(v, v, scale);
        return v;
    }
    template <class type>
    inline vvect_t<type> operator /(const vvect_t<type>& v, type scale) {
        vvect_t<type> val(v.size());
        assign_div(val, v, scale);
        return val;
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
    inline void assign_elem_prod(
        vvect_t<type>& dst, const vvect_t<type>& v1, const vvect_t<type>& v2
    ) {
        assert(dst.size() == v1.size() && v1.size() == v2.size());
        #pragma omp simd
        for (int i = 0; i < dst.size(); i++) {dst[i] = v1[i] * v2[i];}
    }
    template <class type>
    inline vvect_t<type>& apply_elem_prod(vvect_t<type>& v1, const vvect_t<type>& v2) {
        assign_elem_prod(v1, v1, v2);
        return v1;
    }
    template <class type>
    inline vvect_t<type> elem_prod(const vvect_t<type>& v1, const vvect_t<type>& v2) {
        vvect_t<type> val(v1.size());
        assign_elem_prod(val, v1, v2);
        return val;
    }
}
