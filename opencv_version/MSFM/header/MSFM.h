#include <opencv/cv.h>

#ifndef MSFM_H
#define	MSFM_H

class MSFM {
public:
    cv::Mat MSFMSurfaceO1(cv::Mat &image, cv::vector<cv::Point> &initials, cv::Point2d &h);
    cv::Mat MSFMSurfaceO2(cv::Mat &image, cv::vector<cv::Point> &initials, cv::Point2d &h);
    virtual ~MSFM();
private:
    void StencilS1O1(cv::Mat &image, cv::Mat &u_surface, cv::Mat &state, std::multimap<double, cv::Point> &trial_set, 
            std::map<int, std::multimap<double, cv::Point>::iterator> &mapa_trial, cv::Point &neigh, cv::Point2d &h);
    void StencilS2O1(cv::Mat &image, cv::Mat &u_surface, cv::Mat &state, std::multimap<double, cv::Point> &trial_set, 
            std::map<int, std::multimap<double, cv::Point>::iterator> &mapa_trial, cv::Point &neigh, cv::Point2d &h);
    void StencilS1O2(cv::Mat &image, cv::Mat &u_surface, cv::Mat &state, std::multimap<double, cv::Point> &trial_set, 
            std::map<int, std::multimap<double, cv::Point>::iterator> &mapa_trial, cv::Point &neigh, cv::Point2d &h);
    void StencilS2O2(cv::Mat &image, cv::Mat &u_surface, cv::Mat &state, std::multimap<double, cv::Point> &trial_set, 
            std::map<int, std::multimap<double, cv::Point>::iterator> &mapa_trial, cv::Point &neigh, cv::Point2d &h);
};

#endif	/* MSFM_H */

