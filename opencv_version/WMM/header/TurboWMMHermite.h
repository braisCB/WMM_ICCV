#include <opencv/cv.h>

#ifndef TURBOWMMHERMITE_H
#define	TURBOWMMHERMITE_H

typedef struct {
    cv::Point p;
    double v0, v1, v2;
    double m0, m1, m2, m3;
    double fm0, fm1, fm2, fm3;
    int dir;
} TIsoWavefrontH;

class TurboWMMHermite {
public:
    TurboWMMHermite();
    cv::Mat AniSurfaceGrad(cv::Mat &image, cv::vector<cv::Point> &initials, cv::Point2d &h);
    cv::Mat AniSurfaceHL(cv::Mat& image, cv::vector<cv::Point>& initials, cv::Point2d &h);
    cv::Mat AniSurfaceGS(cv::Mat& image, cv::vector<cv::Point>& initials, cv::Point2d &h);
    virtual ~TurboWMMHermite();
private:
    void getHermite(double *y, std::pair<double, double> *m, int pos);
    double Gradient(cv::Mat &image, cv::Mat &u_surface, TIsoWavefrontH &wave, cv::Point &neigh, cv::Point2d &h, bool forced = false);
    double HopfLax(cv::Mat &image, cv::Mat &u_surface, TIsoWavefrontH &wave, cv::Point &neigh, cv::Point2d &h, bool forced = false);
    double GoldenSearch(cv::Mat &image, cv::Mat &u_surface, TIsoWavefrontH &wave, cv::Point &neigh, cv::Point2d &h, bool forced = false);
    int yarray[8];
    int xarray[8];
    bool isnewpos[8];
    double valcenter[8];
    double imcenter[8];
    std::pair<double, double> ms[8];
    std::pair<double, double> ms2[8];
    
};

#endif	/* TURBOWMMHERMITE_H */


