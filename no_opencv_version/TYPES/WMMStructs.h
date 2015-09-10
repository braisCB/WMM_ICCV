#include <stdlib.h>
#include <math.h>

#ifndef MPPSTRUCTS_H
#define	MPPSTRUCTS_H

namespace wmm {

    const int yarray[8] = {-1, -1, -1, 0, 1, 1, 1, 0};
    const int xarray[8] = {-1, 0, 1, 1, 1, 0, -1, -1};

    const int I_LINEAR = 0;
    const int I_HERMITE = 1;
    const int I_PCHIP = 2;
    const int I_SPLINE = 3;
    const int I_QUADATRIC = 4;
    const int M_GRADIENT = 0;
    const int M_HOPFLAX = 1;
    const int M_GOLDENSEARCH = 2;
    const int S_RIGHT = 0;
    const int S_LEFT = 1;
    const unsigned char P_ALIVE = 1;
    const unsigned char P_TRIAL = 2;
    double TAU = 1e-03;
    double MAX_VAL = 100000000;
    double RESPHI = (sqrt(5.0)-1.0)/2.0;

    template<typename _Tp> struct Node_ {
        _Tp y;
        _Tp x;

        inline Node_(_Tp _y = 0, _Tp _x = 0) : y(_y), x(_x) {}

    };

    template<typename _Tp> inline Node_<_Tp> operator+(const Node_<_Tp> &p1, const Node_<_Tp> &p2) {
        Node_<_Tp> p(p1.y + p2.y, p1.x + p2.x);
        return p;
    }

    template<typename _Tp> inline Node_<_Tp> operator-(const Node_<_Tp> &p1, const Node_<_Tp> &p2) {
        Node_<_Tp> p(p1.y - p2.y, p1.x - p2.x);
        return p;
    }

    template<typename _Tp> inline Node_<_Tp> operator*(const _Tp &v, const Node_<_Tp> &p) {
        Node_<_Tp> np(v*p.y, v*p.x);
        return np;
    }

    template<typename _Tp> inline double norm(const Node_<_Tp> &p) {
        return sqrt((double) (p.x*p.x + p.y*p.y));
    }

    typedef Node_<int> Node;
    typedef Node_<double> NodeD;

    template<typename _Tp> inline _Tp* initArray(int anum, _Tp v) {
        _Tp *out = (_Tp*) malloc(anum*sizeof(_Tp));
        for (int i=0; i < anum; i++) {
            out[i] = v;
        }
        return out;
    }

    template<typename _Tp> struct Grid_ {
        int rows;
        int cols;
        int channels;
        _Tp* data;

        inline Grid_(int _rows, int _cols, int _channels=1) : rows(_rows), cols(_cols), channels(_channels), data((_Tp*) calloc(_rows*_cols*_channels,sizeof(_Tp))) {}
        inline Grid_(_Tp* _data, int _rows, int _cols, int _channels=1) : rows(_rows), cols(_cols), channels(_channels), data(_data) {}
        inline Grid_(_Tp _v, int _rows, int _cols, int _channels) : rows(_rows), cols(_cols), channels(_channels), data(initArray(_rows*_cols*_channels, _v)) {}

        inline _Tp& at(Node &p, int dim=0) {
            return data[p.y + rows*(p.x + cols*dim)];
        }

        inline bool contains(Node &p) {
            return (p.x >= 0 && p.y >= 0 && p.x < cols && p.y < rows);
        }

    };

    typedef Grid_<double> Grid;


    template<typename _Tp> struct Wmm0_ {
        Node p;
        _Tp v[3];
        int dir;
    };

    template<typename _Tp, int N> struct Wmm_ : public Wmm0_<_Tp> {
        _Tp m[N];
        _Tp fm[N];
    };

    template<typename _Tp> struct Wmm_<_Tp, 0> : public Wmm0_<_Tp> {
        _Tp m[1];
        _Tp fm[1];
    };

    typedef Wmm_<double, 4> WmmNode;

}
#endif	/* MPPSTRUCTS_H */

