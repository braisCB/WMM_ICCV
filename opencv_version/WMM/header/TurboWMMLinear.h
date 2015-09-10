#include <opencv/cv.h>

#ifndef TURBOWMMLINEAR_H
#define	TURBOWMMLINEAR_H

typedef struct {
    cv::Point p;
    double v0, v1, v2;
    int dir;
} TIsoWavefrontL;

class TurboWMMLinear {
public:
    TurboWMMLinear();
    cv::Mat AniSurfaceGrad(cv::Mat &image, cv::vector<cv::Point> &initials, cv::Point2d &h);
    cv::Mat AniSurfaceHL(cv::Mat& image, cv::vector<cv::Point>& initials, cv::Point2d &h);
    cv::Mat AniSurfaceGS(cv::Mat& image, cv::vector<cv::Point>& initials, cv::Point2d &h);
    virtual ~TurboWMMLinear();
private:
    double Gradient(cv::Mat &image, cv::Mat &u_surface, TIsoWavefrontL &wave, cv::Point &neigh, cv::Point2d &h, bool forced = false);
    double HopfLax(cv::Mat &image, cv::Mat &u_surface, TIsoWavefrontL &wave, cv::Point &neigh, cv::Point2d &h, bool forced = false);
    double GoldenSearch(cv::Mat &image, cv::Mat &u_surface, TIsoWavefrontL &wave, cv::Point &neigh, cv::Point2d &h, bool forced = false);
    int yarray[8];
    int xarray[8];
    bool isnewpos[8];
    double valcenter[8];
    
    
};

#endif	/* TURBOWMMLINEAR_H */


