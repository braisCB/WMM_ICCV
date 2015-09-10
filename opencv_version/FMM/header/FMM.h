#include <opencv/cv.h>

#ifndef FMM_H
#define	FMM_H

class FMM {
public:
    cv::Mat FMMSurfaceO1(cv::Mat &image, cv::vector<cv::Point> &initials, cv::Point2d &h);
    cv::Mat FMMSurfaceO2(cv::Mat &image, cv::vector<cv::Point> &initials, cv::Point2d &h);
    virtual ~FMM();
private:
    void StencilS1O1(cv::Mat &image, cv::Mat &u_surface, cv::Mat &state, std::multimap<double, cv::Point> &trial_set, 
            std::map<int, std::multimap<double, cv::Point>::iterator> &mapa_trial, cv::Point &neigh, cv::Point2d &h);
    void StencilS1O2(cv::Mat &image, cv::Mat &u_surface, cv::Mat &state, std::multimap<double, cv::Point> &trial_set, 
            std::map<int, std::multimap<double, cv::Point>::iterator> &mapa_trial, cv::Point &neigh, cv::Point2d &h);
};

#endif	/* FMM_H */

