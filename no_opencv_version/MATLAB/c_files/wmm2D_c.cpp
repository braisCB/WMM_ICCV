/* GCMex.cpp Version 2.3.0 
 *
 * Copyright 2009 Brian Fulkerson <bfulkers@cs.ucla.edu> 
 */

#include "mex.h"
#include "../../2D/wave_mm_2D.cpp"
#include "../../TYPES/WMMStructs.h"
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <string.h>

int getWMMdegree(int interp) {
    switch (interp) {
        case wmm::I_LINEAR:
            return 0;
        case wmm::I_QUADATRIC:
            return 5;
        default: //I_SPLINE, I_HERMITE and I_PCHIP
            return 4;
    }
}

void mexFunction(
        int		  nout, 	/* number of expected outputs */
        mxArray	  *out[],	/* mxArray output pointer array */
        int		  nin, 	/* number of inputs */
        const mxArray	  *in[]	/* mxArray input pointer array */
)
{

    enum {IN_IMAGE=0,IN_INITIALS,IN_H,IN_INTERP,IN_MODE} ;
    enum {OUT_SURFACE=0} ;

    /****************************************************************************
    * ERROR CHECKING
    ***************************************************************************/

    if (nin != 5)
        mexErrMsgTxt("Incorrect number of arguments. See 'help wmm2D'");

    if(mxGetNumberOfDimensions(in[IN_IMAGE]) != 3 ||
            mxGetClassID(in[IN_IMAGE]) != mxDOUBLE_CLASS)
        mexErrMsgTxt("Potential must be a MxNx2 matrix of class DOUBLE. See 'help wmm2D'");

    int num_points = mxGetM(in[IN_INITIALS]);
    int num_param = mxGetN(in[IN_INITIALS]);

    if (num_points == 0 || num_param != 2 ||
            mxGetClassID(in[IN_INITIALS]) != mxDOUBLE_CLASS)
        mexErrMsgTxt("Initials should be a Lx2 matrix. See 'help wmm2D'");

    double nh = mxGetN(in[IN_H]);
    if (mxGetM(in[IN_H]) != 1 || nh != 2 ||
            mxGetClassID(in[IN_H]) != mxDOUBLE_CLASS)
        mexErrMsgTxt("h should be an 1x2 vector. See 'help wmm2D'");

    double* hp = mxGetPr(in[IN_H]);
    wmm::NodeD hs = wmm::NodeD(hp[0], hp[1]);

    if (hs.x < 0.0 || hs.y < 0.0)
        mexErrMsgTxt("h should be positive. See 'help wmm2D'");

    size_t K = mxGetNumberOfDimensions(in[IN_IMAGE]);
    const mwSize *N = mxGetDimensions(in[IN_IMAGE]);

    if (N[2] != 2)
        mexErrMsgTxt("Potential must be a MxNx2 matrix of class DOUBLE. See 'help wmm2D'");


    char *c_search;
    c_search = mxArrayToString(in[IN_MODE]);
    int search;
    if (strcmp(c_search, "gradient") == 0)
        search = wmm::M_GRADIENT;
    else if (strcmp(c_search, "hopf_lax") == 0)
        search = wmm::M_HOPFLAX;
    else if (strcmp(c_search, "golden_search") == 0)
        search = wmm::M_GOLDENSEARCH;
    else {
        mxFree(c_search);
        mexErrMsgTxt("Unknown searching method. See 'help wmm2D'");
    }
    mxFree(c_search);


    char *c_interp;
    c_interp = mxArrayToString(in[IN_INTERP]);
    int interp;
    if (strcmp(c_interp, "linear") == 0)
        interp = wmm::I_LINEAR;
    else if (strcmp(c_interp, "quad") == 0)
        interp = wmm::I_QUADATRIC;
    else if (strcmp(c_interp, "spline") == 0)
        interp = wmm::I_SPLINE;
    else  if (strcmp(c_interp, "pchip") == 0)
        interp = wmm::I_PCHIP;
    else  if (strcmp(c_interp, "hermite") == 0)
        interp = wmm::I_HERMITE;
    else {
        mxFree(c_interp);
        mexErrMsgTxt("Unknown interpolation method. See 'help wmm2D'");
    }
    mxFree(c_interp);


    int rows = N[0];
    int cols = N[1];

    /* Create output arrays */
    mwSize dims[2] = {rows,cols};
    out[OUT_SURFACE] = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
    double * surface = mxGetPr(out[OUT_SURFACE]);


    /* Data costs are nlabels rows x npixels cols */
    double* inits = mxGetPr(in[IN_INITIALS]);


    wmm::Grid imagen(mxGetPr(in[IN_IMAGE]), rows, cols);

    std::vector<wmm::Node> initials;
    wmm::Node p;
    for (int i = 0; i < num_points; i++) {
        p = wmm::Node(round(inits[i] - 1), round(inits[i+num_points] - 1));
        initials.push_back(p);
        if (!imagen.contains(p)) {
            mexErrMsgTxt("Initial point out of the scope");
        }
    }

    wmm::Grid out_surface(mxGetPr(in[IN_IMAGE]), rows, cols);
    switch (interp) {
        case wmm::I_LINEAR:
            out_surface = wmm::WmmIsoSurface2D<0>(imagen, initials, hs, interp, search);
            break;
        case wmm::I_QUADATRIC:
            out_surface = wmm::WmmIsoSurface2D<5>(imagen, initials, hs, interp, search);
            break;
        default: //I_SPLINE, I_HERMITE and I_PCHIP
            out_surface = wmm::WmmIsoSurface2D<4>(imagen, initials, hs, interp, search);
    }

    std::copy(out_surface.data, out_surface.data + rows*cols, surface);

    return;

}