#include <opencv/cv.h>

#ifndef TURBOWMMINTERP_H
#define	TURBOWMMINTERP_H

typedef struct {
    cv::Point p;
    double v0, v1, v2;
    double m0, m1, m2, m3, m4, m5;
    double fm0, fm1, fm2, fm3, fm4, fm5;
    int dir;
} TIsoWavefrontI;

typedef struct {
    double p0, p1, p2;
} TripleI;

class TurboWMMInterp {
public:
    TurboWMMInterp();
    cv::Mat AniSurfaceGrad(cv::Mat &image, cv::vector<cv::Point> &initials, cv::Point2d &h);
    cv::Mat AniSurfaceHL(cv::Mat& image, cv::vector<cv::Point>& initials, cv::Point2d &h);
    cv::Mat AniSurfaceGS(cv::Mat& image, cv::vector<cv::Point>& initials, cv::Point2d &h);
    virtual ~TurboWMMInterp();
private:
    void getInterp(double *y, TripleI *m, int pos);
    void getInterpInverse(double *y, TripleI *m, int pos);
    double Gradient(cv::Mat &image, cv::Mat &u_surface, TIsoWavefrontI &wave, cv::Point &neigh, cv::Point2d &h, bool forced = false);
    double HopfLax(cv::Mat &image, cv::Mat &u_surface, TIsoWavefrontI &wave, cv::Point &neigh, cv::Point2d &h, bool forced = false);
    double GoldenSearch(cv::Mat &image, cv::Mat &u_surface, TIsoWavefrontI &wave, cv::Point &neigh, cv::Point2d &h, bool forced = false);
    
    int solveLinear(double *C, double *S);
    int solveQuadric(double *C, double *S);
    int solveCubic(double *C, double *S);
    int solveQuartic(double *C, double *S);
    
    int yarray[8];
    int xarray[8];
    bool isnewpos[8];
    double valcenter[8];
    double imcenter[8];
    TripleI ms[8];
    TripleI ms2[8];
    TripleI ms3[8];
    TripleI ms4[8];
};

#endif	/* TURBOWMMINTERP_H */


