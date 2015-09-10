#include "mex.h"
#include "../../../FMM/source/FMM.cpp"
#include <stdlib.h>
#include <opencv/cv.h>
#include <vector>
#include <iostream>

void mexFunction(
    int		  nout, 	/* number of expected outputs */
    mxArray	  *out[],	/* mxArray output pointer array */
    int		  nin, 	/* number of inputs */
    const mxArray	  *in[]	/* mxArray input pointer array */
    )
{
   
  enum {IN_IMAGE=0,IN_INITIALS,IN_H,IN_ORDER} ;
  enum {OUT_SURFACE=0} ;

  /****************************************************************************
   * ERROR CHECKING
   ***************************************************************************/
  
  if (nin != 4)
    mexErrMsgTxt("Incorrect number of arguments. See 'help msmf2D'");

  if(mxGetNumberOfDimensions(in[IN_IMAGE]) != 2 || 
          mxGetClassID(in[IN_IMAGE]) != mxDOUBLE_CLASS)
    mexErrMsgTxt("Image must be a MxN matrix of class DOUBLE. See 'help msmf2D'");
  
  int num_points = mxGetM(in[IN_INITIALS]);
  int num_param = mxGetN(in[IN_INITIALS]);
  
  if (num_points == 0 || num_param != 2 || 
          mxGetClassID(in[IN_INITIALS]) != mxDOUBLE_CLASS)
      mexErrMsgTxt("Initials should be a Lx2 matrix. See 'help msmf2D'");
  
  double nh = mxGetN(in[IN_H]);
  if (mxGetM(in[IN_H]) != 1 || nh != 2 || 
          mxGetClassID(in[IN_H]) != mxDOUBLE_CLASS)
      mexErrMsgTxt("h should be an 1x2 vector. See 'help wmm2D'");
  
  double* hp = mxGetPr(in[IN_H]);
  cv::Point2d hs;
  hs = cv::Point2d(hp[1], hp[0]);
  
  if (hs.x < 0.0 || hs.y < 0.0)
      mexErrMsgTxt("h should be positive. See 'help msfm2D'");
  
  if (mxGetM(in[IN_ORDER]) != 1 || mxGetN(in[IN_ORDER]) != 1 || 
          mxGetClassID(in[IN_ORDER]) != mxDOUBLE_CLASS)
      mexErrMsgTxt("order should be 1 or 2. See 'help msmf2D'");

  int order = (int) mxGetScalar(in[IN_ORDER]);
  
  if (order < 1 || order > 2)
      mexErrMsgTxt("order should be 1 or 2. See 'help msmf2D'");
  
  size_t K = mxGetNumberOfDimensions(in[IN_IMAGE]);
  const mwSize *N = mxGetDimensions(in[IN_IMAGE]);
  
  int rows = N[0];
  int cols = N[1];

  /* Create output arrays */
  mwSize dims[2] = {rows,cols};
  out[OUT_SURFACE] = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
  double * surface = mxGetPr(out[OUT_SURFACE]);
  
  
  /* Data costs are nlabels rows x npixels cols */
  double* inits = mxGetPr(in[IN_INITIALS]);

  cv::Rect rect(0, 0, rows, cols);
  
  cv::Mat imagen;
  imagen = cv::Mat(rows, cols, CV_64FC1, mxGetPr(in[IN_IMAGE]));
  
  cv::vector<cv::Point> initials;
  cv::Point p;
  for (int i = 0; i < num_points; i++) {
      p = cv::Point(round(inits[i+num_points] - 1), round(inits[i] - 1));
      initials.push_back(p);
      if (!rect.contains(p)) {
          mexErrMsgTxt("Initial point out of the scope");
      }
  }
  
  cv::Mat out_surface;
  FMM fmm;
  if (order == 1)
      out_surface = fmm.FMMSurfaceO1(imagen, initials, hs);
  else
      out_surface = fmm.FMMSurfaceO2(imagen, initials, hs);
  
  double *odata = reinterpret_cast<double *>(out_surface.data);
  
  std::copy(odata, odata + rows*cols, surface);
  return;

}
