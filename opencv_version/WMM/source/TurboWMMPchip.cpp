#include "../header/TurboWMMPchip.h"
#include <opencv/cv.h>
#include <map>

#define MAX_VAL 100000
#define P_FAR   0
#define P_ALIVE 1
#define P_TRIAL 2
#define RESPHI  (sqrt(5.0)-1.0)/2.0
#define TAU     1e-03

#define timageP(p,k) reinterpret_cast<double *>(image.data)[(int) (p.y + image.rows*(p.x + image.cols*k))]
#define tdistanceP(p) reinterpret_cast<double *>(u_surface.data)[(int) (p.y + u_surface.rows*p.x)]

TurboWMMPchip::TurboWMMPchip() {
    this->yarray[0] = -1; this->yarray[1] = -1; this->yarray[2] = -1; this->yarray[3] = 0;
    this->yarray[4] = 1; this->yarray[5] = 1; this->yarray[6] = 1; this->yarray[7] = 0;
    this->xarray[0] = -1; this->xarray[1] = 0; this->xarray[2] = 1; this->xarray[3] = 1;
    this->xarray[4] = 1; this->xarray[5] = 0; this->xarray[6] = -1; this->xarray[7] = -1;
}

cv::Mat TurboWMMPchip::AniSurfaceGrad(cv::Mat& image, cv::vector<cv::Point>& initials, cv::Point2d &h) {

    cv::Mat u_surface = MAX_VAL*cv::Mat::ones(image.rows, image.cols, CV_64FC1);
    cv::Mat state = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
    
    std::multimap<double, TIsoWavefrontP> trial_set;
    std::map<int, std::multimap<double, TIsoWavefrontP>::iterator> mapa_trial;
    
    std::multimap<double, TIsoWavefrontP>::iterator trial_set_it;
    std::map<int, std::multimap<double, TIsoWavefrontP>::iterator>::iterator mapa_trial_it;
    std::pair<double, TIsoWavefrontP> pr_trial;
    std::pair<int, std::multimap<double, TIsoWavefrontP>::iterator> pr_mapa;
    int key, i;
    TIsoWavefrontP winner, new_w;
    cv::Point neigh;
            
    cv::Rect imagerect(0, 0, image.cols, image.rows);
    
    // Initialization
    for (i = 0; i < (int) initials.size(); i++) {
        key = initials[i].y*u_surface.cols + initials[i].x;
        if (mapa_trial.find(key) == mapa_trial.end() && imagerect.contains(initials[i])) {
            tdistanceP(initials[i]) = 0.0;
            winner.dir = -1;
            winner.v0 = 0.0;
            winner.p = initials[i];
            state.at<unsigned char>(initials[i]) = P_TRIAL;
            pr_trial = std::pair<double, TIsoWavefrontP >(0.0, winner);
            trial_set_it = trial_set.insert(pr_trial);
            pr_mapa = std::pair<int, std::multimap<double, TIsoWavefrontP >::iterator>(key, trial_set_it);
            mapa_trial.insert(pr_mapa);
        }
    }
    
    while (!trial_set.empty()) {
        
        trial_set_it = trial_set.begin();
        key = trial_set_it->second.p.y*u_surface.cols + trial_set_it->second.p.x;
        mapa_trial_it = mapa_trial.find(key);
                
        if (mapa_trial_it == mapa_trial.end()) {
            printf("ERROR: bad map alloc");
            exit(-1);
        }
        
        if (mapa_trial_it->second != trial_set_it) {
            printf("ERROR: bad trial/map alloc");
            exit(-1);
        }
        
        winner = trial_set_it->second;
        
        trial_set.erase(trial_set_it);
        mapa_trial.erase(mapa_trial_it);
        
        state.at<unsigned char>(winner.p) = P_ALIVE;

        for (int i=0; i < 8; i++) {
            neigh = winner.p + cv::Point(this->xarray[i], this->yarray[i]);
            isnewpos[i] = false;
            this->valcenter[i] = imagerect.contains(neigh) ? tdistanceP(neigh) : MAX_VAL;
            this->imcenter[i] = imagerect.contains(neigh) ? cv::norm(cv::Point2d(timageP(neigh,0), timageP(neigh,1))) : MAX_VAL;
            if (imagerect.contains(neigh) && state.at<unsigned char>(neigh) != P_ALIVE) {
                double val_neigh = this->Gradient(image, u_surface, winner, neigh, h);
                if (val_neigh < this->valcenter[i]) {
                    this->valcenter[i] = val_neigh;
                    this->isnewpos[i] = true;
                }
            }
        }
        
        for (int i=0; i < 8; i++) {
            if (this->isnewpos[i]) {
                new_w.p = cv::Point(winner.p.x + this->xarray[i], winner.p.y + this->yarray[i]);
                if (this->valcenter[i] <= winner.v0  && this->valcenter[i] <= this->valcenter[(i+1)%8] && this->valcenter[i] <= this->valcenter[(i+7)%8] && 
                    (i%2 == 0 || (this->valcenter[i] <= this->valcenter[(i+2)%8] && this->valcenter[i] <= this->valcenter[(i+6)%8]))) {
                    double val_neigh = this->Gradient(image, u_surface, winner, new_w.p, h, true);
                    this->valcenter[i] = tdistanceP(new_w.p);
                    if (val_neigh < this->valcenter[i]) {
                        this->isnewpos[i] = true;
                        this->valcenter[i] = val_neigh;
                    }
                }
            }
        }
        
        // Pchips
        for (int i=0; i < 8; i+=2) {
            this->getPCHIP(this->valcenter, this->ms, i);
            this->getPCHIP(this->imcenter, this->ms2, i);
        }
        
        // Update
        for (int i=0; i < 8; i++) {
            if (isnewpos[i]) {
                neigh = winner.p + cv::Point(this->xarray[i], this->yarray[i]);
                key = neigh.y*u_surface.cols + neigh.x;
                if (state.at<unsigned char>(neigh) == P_TRIAL) {
                    mapa_trial_it = mapa_trial.find(key);
                    trial_set.erase(mapa_trial_it->second);
                    mapa_trial.erase(mapa_trial_it);
                }
                else {
                    state.at<unsigned char>(neigh) = P_TRIAL;
                }
                new_w.p = neigh;
                new_w.dir = i;
                new_w.v0 = this->valcenter[i];
                new_w.v1 = this->valcenter[(i+1)%8];
                new_w.v2 = this->valcenter[(i+7)%8];
                new_w.m0 = this->ms[i].first;
                new_w.m1 = this->ms[i].second;
                new_w.m2 = -this->ms[(i+7)%8].second;
                new_w.m3 = -this->ms[(i+7)%8].first;
                new_w.fm0 = this->ms2[i].first;
                new_w.fm1 = this->ms2[i].second;
                new_w.fm2 = -this->ms2[(i+7)%8].second;
                new_w.fm3 = -this->ms2[(i+7)%8].first;
                
                pr_trial = std::pair<double, TIsoWavefrontP>(valcenter[i], new_w);
                trial_set_it = trial_set.insert(pr_trial);
                pr_mapa = std::pair<int, std::multimap<double, TIsoWavefrontP>::iterator>(key, trial_set_it);
                mapa_trial.insert(pr_mapa);
                                
                tdistanceP(new_w.p) = valcenter[i];
            }
        }
        
    }
    
    return u_surface;
    
}


cv::Mat TurboWMMPchip::AniSurfaceHL(cv::Mat& image, cv::vector<cv::Point>& initials, cv::Point2d &h) {

    cv::Mat u_surface = MAX_VAL*cv::Mat::ones(image.rows, image.cols, CV_64FC1);
    cv::Mat state = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
    
    std::multimap<double, TIsoWavefrontP> trial_set;
    std::map<int, std::multimap<double, TIsoWavefrontP>::iterator> mapa_trial;
    
    std::multimap<double, TIsoWavefrontP>::iterator trial_set_it;
    std::map<int, std::multimap<double, TIsoWavefrontP>::iterator>::iterator mapa_trial_it;
    std::pair<double, TIsoWavefrontP> pr_trial;
    std::pair<int, std::multimap<double, TIsoWavefrontP>::iterator> pr_mapa;
    int key, i;
    TIsoWavefrontP winner, new_w;
    cv::Point neigh;
            
    cv::Rect imagerect(0, 0, image.cols, image.rows);
    
    // Initialization
    for (i = 0; i < (int) initials.size(); i++) {
        key = initials[i].y*u_surface.cols + initials[i].x;
        if (mapa_trial.find(key) == mapa_trial.end() && imagerect.contains(initials[i])) {
            tdistanceP(initials[i]) = 0.0;
            winner.dir = -1;
            winner.v0 = 0.0;
            winner.p = initials[i];
            state.at<unsigned char>(initials[i]) = P_TRIAL;
            pr_trial = std::pair<double, TIsoWavefrontP >(0.0, winner);
            trial_set_it = trial_set.insert(pr_trial);
            pr_mapa = std::pair<int, std::multimap<double, TIsoWavefrontP >::iterator>(key, trial_set_it);
            mapa_trial.insert(pr_mapa);
        }
    }
    
    while (!trial_set.empty()) {
        
        trial_set_it = trial_set.begin();
        key = trial_set_it->second.p.y*u_surface.cols + trial_set_it->second.p.x;
        mapa_trial_it = mapa_trial.find(key);
                
        if (mapa_trial_it == mapa_trial.end()) {
            printf("ERROR: bad map alloc");
            exit(-1);
        }
        
        if (mapa_trial_it->second != trial_set_it) {
            printf("ERROR: bad trial/map alloc");
            exit(-1);
        }
        
        winner = trial_set_it->second;
        
//        std::cout << "------ GANADOR " << winner.p << " ----> " << winner.v0 << ", " << winner.v1 << ", " << winner.v2 << std::endl;
//        std::cout << "------    PCHIP " << winner.m0 << ", " << winner.m1 << ", " << winner.m2 << ", " << winner.m3 << std::endl;
//        std::cout << "------   FPCHIP " << winner.fm0 << ", " << winner.fm1 << ", " << winner.fm2 << ", " << winner.fm3 << std::endl;
        
        trial_set.erase(trial_set_it);
        mapa_trial.erase(mapa_trial_it);
        
        state.at<unsigned char>(winner.p) = P_ALIVE;

        for (int i=0; i < 8; i++) {
            neigh = winner.p + cv::Point(this->xarray[i], this->yarray[i]);
            isnewpos[i] = false;
            this->valcenter[i] = imagerect.contains(neigh) ? tdistanceP(neigh) : MAX_VAL;
            this->imcenter[i] = imagerect.contains(neigh) ? cv::norm(cv::Point2d(timageP(neigh,0), timageP(neigh,1))) : MAX_VAL;
            if (imagerect.contains(neigh) && state.at<unsigned char>(neigh) != P_ALIVE) {
                double val_neigh = this->HopfLax(image, u_surface, winner, neigh, h);
                if (val_neigh < this->valcenter[i]) {
                    this->valcenter[i] = val_neigh;
                    this->isnewpos[i] = true;
                }
            }
        }
        
        for (int i=0; i < 8; i++) {
            if (this->isnewpos[i]) {
                new_w.p = cv::Point(winner.p.x + this->xarray[i], winner.p.y + this->yarray[i]);
                if (this->valcenter[i] <= winner.v0  && this->valcenter[i] <= this->valcenter[(i+1)%8] && this->valcenter[i] <= this->valcenter[(i+7)%8] && 
                    (i%2 == 0 || (this->valcenter[i] <= this->valcenter[(i+2)%8] && this->valcenter[i] <= this->valcenter[(i+6)%8]))) {
                    double val_neigh = this->HopfLax(image, u_surface, winner, new_w.p, h, true);
                    this->valcenter[i] = tdistanceP(new_w.p);
                    if (val_neigh < this->valcenter[i]) {
                        this->isnewpos[i] = true;
                        this->valcenter[i] = val_neigh;
                    }
                }
            }
        }
        
        // Pchips
        for (int i=0; i < 8; i+=2) {
            this->getPCHIP(this->valcenter, this->ms, i);
            this->getPCHIP(this->imcenter, this->ms2, i);
        }
        
        // Update
        for (int i=0; i < 8; i++) {
            if (isnewpos[i]) {
                neigh = winner.p + cv::Point(this->xarray[i], this->yarray[i]);
                key = neigh.y*u_surface.cols + neigh.x;
                if (state.at<unsigned char>(neigh) == P_TRIAL) {
                    mapa_trial_it = mapa_trial.find(key);
                    trial_set.erase(mapa_trial_it->second);
                    mapa_trial.erase(mapa_trial_it);
                }
                else {
                    state.at<unsigned char>(neigh) = P_TRIAL;
                }
                new_w.p = neigh;
                new_w.dir = i;
                new_w.v0 = this->valcenter[i];
                new_w.v1 = this->valcenter[(i+1)%8];
                new_w.v2 = this->valcenter[(i+7)%8];
                new_w.m0 = this->ms[i].first;
                new_w.m1 = this->ms[i].second;
                new_w.m2 = -this->ms[(i+7)%8].second;
                new_w.m3 = -this->ms[(i+7)%8].first;
                new_w.fm0 = this->ms2[i].first;
                new_w.fm1 = this->ms2[i].second;
                new_w.fm2 = -this->ms2[(i+7)%8].second;
                new_w.fm3 = -this->ms2[(i+7)%8].first;
                
                pr_trial = std::pair<double, TIsoWavefrontP>(valcenter[i], new_w);
                trial_set_it = trial_set.insert(pr_trial);
                pr_mapa = std::pair<int, std::multimap<double, TIsoWavefrontP>::iterator>(key, trial_set_it);
                mapa_trial.insert(pr_mapa);
                                
                tdistanceP(new_w.p) = valcenter[i];
                
                //std::cout << "ACTUALIZANDO " << new_w.p << " ----> " << new_w.v0 << std::endl;
            }
        }
        
    }
    
    return u_surface;
    
}


cv::Mat TurboWMMPchip::AniSurfaceGS(cv::Mat& image, cv::vector<cv::Point>& initials, cv::Point2d &h) {

    cv::Mat u_surface = MAX_VAL*cv::Mat::ones(image.rows, image.cols, CV_64FC1);
    cv::Mat state = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
    
    std::multimap<double, TIsoWavefrontP> trial_set;
    std::map<int, std::multimap<double, TIsoWavefrontP>::iterator> mapa_trial;
    
    std::multimap<double, TIsoWavefrontP>::iterator trial_set_it;
    std::map<int, std::multimap<double, TIsoWavefrontP>::iterator>::iterator mapa_trial_it;
    std::pair<double, TIsoWavefrontP> pr_trial;
    std::pair<int, std::multimap<double, TIsoWavefrontP>::iterator> pr_mapa;
    int key, i;
    TIsoWavefrontP winner, new_w;
    cv::Point neigh;
            
    cv::Rect imagerect(0, 0, image.cols, image.rows);
    
    // Initialization
    for (i = 0; i < (int) initials.size(); i++) {
        key = initials[i].y*u_surface.cols + initials[i].x;
        if (mapa_trial.find(key) == mapa_trial.end() && imagerect.contains(initials[i])) {
            tdistanceP(initials[i]) = 0.0;
            winner.dir = -1;
            winner.v0 = 0.0;
            winner.p = initials[i];
            state.at<unsigned char>(initials[i]) = P_TRIAL;
            pr_trial = std::pair<double, TIsoWavefrontP >(0.0, winner);
            trial_set_it = trial_set.insert(pr_trial);
            pr_mapa = std::pair<int, std::multimap<double, TIsoWavefrontP >::iterator>(key, trial_set_it);
            mapa_trial.insert(pr_mapa);
        }
    }
    
    while (!trial_set.empty()) {
        
        trial_set_it = trial_set.begin();
        key = trial_set_it->second.p.y*u_surface.cols + trial_set_it->second.p.x;
        mapa_trial_it = mapa_trial.find(key);
                
        if (mapa_trial_it == mapa_trial.end()) {
            printf("ERROR: bad map alloc");
            exit(-1);
        }
        
        if (mapa_trial_it->second != trial_set_it) {
            printf("ERROR: bad trial/map alloc");
            exit(-1);
        }
        
        winner = trial_set_it->second;
        
        trial_set.erase(trial_set_it);
        mapa_trial.erase(mapa_trial_it);
        
        state.at<unsigned char>(winner.p) = P_ALIVE;

        for (int i=0; i < 8; i++) {
            neigh = winner.p + cv::Point(this->xarray[i], this->yarray[i]);
            isnewpos[i] = false;
            this->valcenter[i] = imagerect.contains(neigh) ? tdistanceP(neigh) : MAX_VAL;
            this->imcenter[i] = imagerect.contains(neigh) ? cv::norm(cv::Point2d(timageP(neigh,0), timageP(neigh,1))) : MAX_VAL;
            if (imagerect.contains(neigh) && state.at<unsigned char>(neigh) != P_ALIVE) {
                double val_neigh = this->GoldenSearch(image, u_surface, winner, neigh, h);
                if (val_neigh < this->valcenter[i]) {
                    this->valcenter[i] = val_neigh;
                    this->isnewpos[i] = true;
                }
            }
        }
        
        for (int i=0; i < 8; i++) {
            if (this->isnewpos[i]) {
                new_w.p = cv::Point(winner.p.x + this->xarray[i], winner.p.y + this->yarray[i]);
                if (this->valcenter[i] <= winner.v0  && this->valcenter[i] <= this->valcenter[(i+1)%8] && this->valcenter[i] <= this->valcenter[(i+7)%8] && 
                    (i%2 == 0 || (this->valcenter[i] <= this->valcenter[(i+2)%8] && this->valcenter[i] <= this->valcenter[(i+6)%8]))) {
                    double val_neigh = this->GoldenSearch(image, u_surface, winner, new_w.p, h, true);
                    this->valcenter[i] = tdistanceP(new_w.p);
                    if (val_neigh < this->valcenter[i]) {
                        this->isnewpos[i] = true;
                        this->valcenter[i] = val_neigh;
                    }
                }
            }
        }
        
        // Pchips
        for (int i=0; i < 8; i+=2) {
            this->getPCHIP(this->valcenter, this->ms, i);
            this->getPCHIP(this->imcenter, this->ms2, i);
        }
        
        // Update
        for (int i=0; i < 8; i++) {
            if (isnewpos[i]) {
                neigh = winner.p + cv::Point(this->xarray[i], this->yarray[i]);
                key = neigh.y*u_surface.cols + neigh.x;
                if (state.at<unsigned char>(neigh) == P_TRIAL) {
                    mapa_trial_it = mapa_trial.find(key);
                    trial_set.erase(mapa_trial_it->second);
                    mapa_trial.erase(mapa_trial_it);
                }
                else {
                    state.at<unsigned char>(neigh) = P_TRIAL;
                }
                new_w.p = neigh;
                new_w.dir = i;
                new_w.v0 = this->valcenter[i];
                new_w.v1 = this->valcenter[(i+1)%8];
                new_w.v2 = this->valcenter[(i+7)%8];
                new_w.m0 = this->ms[i].first;
                new_w.m1 = this->ms[i].second;
                new_w.m2 = -this->ms[(i+7)%8].second;
                new_w.m3 = -this->ms[(i+7)%8].first;
                new_w.fm0 = this->ms2[i].first;
                new_w.fm1 = this->ms2[i].second;
                new_w.fm2 = -this->ms2[(i+7)%8].second;
                new_w.fm3 = -this->ms2[(i+7)%8].first;
                
                pr_trial = std::pair<double, TIsoWavefrontP>(valcenter[i], new_w);
                trial_set_it = trial_set.insert(pr_trial);
                pr_mapa = std::pair<int, std::multimap<double, TIsoWavefrontP>::iterator>(key, trial_set_it);
                mapa_trial.insert(pr_mapa);
                                
                tdistanceP(new_w.p) = valcenter[i];
            }
        }
        
    }
    
    return u_surface;
    
}


double TurboWMMPchip::Gradient(cv::Mat &image, cv::Mat &u_surface, TIsoWavefrontP &wave, cv::Point &neigh, cv::Point2d &h, bool forced) {
    
    cv::Point2d f0(timageP(wave.p, 1), timageP(wave.p, 0)), fn(timageP(neigh, 1), timageP(neigh, 0));
    double y0 = wave.v0;
    
    if (std::isinf(cv::norm(f0)) || std::isnan(cv::norm(f0)))
        f0 = fn;
    
    double val = MAX_VAL;
    if (wave.dir < 0) {
        cv::Point2d diff(h.x*(neigh.x - wave.p.x), h.y*(neigh.y - wave.p.y));
        val = y0 + cv::norm(diff)*(cv::norm(f0) + cv::norm(fn))/2.0;
    }
    else {
        cv::Rect imagerect(0, 0, image.cols, image.rows);
        
        cv::Point p(wave.p.x + this->xarray[(wave.dir+1)%8] - this->xarray[wave.dir], 
                wave.p.y + this->yarray[(wave.dir+1)%8] - this->yarray[wave.dir]);
        double res1 = MAX_VAL;
        cv::Point2d dp(h.x*wave.p.x, h.y*wave.p.y), dn(h.x*neigh.x, h.y*neigh.y);
        
        if (imagerect.contains(p)) {
            double y1 = wave.v1;
            
            cv::Point2d dd(h.x*(this->xarray[(wave.dir+1)%8] - this->xarray[wave.dir]), h.y*(this->yarray[(wave.dir+1)%8] - this->yarray[wave.dir]));

            if (forced && cv::norm(dn - dp - dd) - cv::norm(h) > TAU ) {
                res1 = y0 + cv::norm(dn - dp)*(cv::norm(f0) + cv::norm(fn))/2.0;
            }
            else {
            
                cv::Point2d f1(timageP(p, 1), timageP(p, 0));
                if (std::isinf(cv::norm(f1)) || std::isnan(cv::norm(f1)))
                    f1 = fn;

                double A = -dd.y, B = dd.x, C = dd.y*dp.x - dd.x*dp.y;
                double den = A*fn.x + B*fn.y;
                double t = (A*dn.x + B*dn.y + C)/den, epsilon;

                cv::Point2d x(dn.x - t*fn.x, dn.y - t*fn.y); 

                if (fabs(dd.x) > 0.0 && fabs(den) > 0.0)
                    epsilon = (x.x - dp.x)/dd.x;
                else if (fabs(dd.y) > 0.0 && fabs(den) > 0.0)
                    epsilon = (x.y - dp.y)/dd.y;
                else if (fabs(den) == 0.0 && cv::norm(dd) > 0.0) {
                    double dist = fabs(A*dn.x + B*dn.y + C)/sqrt(A*A + B*B);
                    epsilon = (cv::norm(dn - dp) - dist)/(fabs(dd.x) + fabs(dd.y)); 
                }
                else
                    epsilon = 0.0;

                if (epsilon < 0.0)
                    epsilon = 0.0;
                else if (epsilon > 1.0)
                    epsilon = 1.0;

                cv::Point2d wp1 = dp + epsilon*dd;
                double t_2 = epsilon*epsilon;
                double t_3 = epsilon*t_2;

                double ft = (2.0*t_3 - 3.0*t_2 + 1.0)*cv::norm(f0) +
                        (t_3 - 2.0*t_2 + epsilon)*wave.fm0 +
                        (-2.0*t_3 + 3.0*t_2)*cv::norm(f1) +
                        (t_3 - t_2)*wave.fm1;

                res1 = (2.0*t_3 - 3.0*t_2 + 1.0)*y0 +
                        (t_3 - 2.0*t_2 + epsilon)*wave.m0 +
                        (-2.0*t_3 + 3.0*t_2)*y1 +
                        (t_3 - t_2)*wave.m1 +
                        cv::norm(dn - wp1)*(ft+cv::norm(fn))/2.0;
            }
        }
        
        p = cv::Point(wave.p.x + this->xarray[(wave.dir+7)%8] - this->xarray[wave.dir], 
                wave.p.y + this->yarray[(wave.dir+7)%8] - this->yarray[wave.dir]);
        double res2 = MAX_VAL;
        
        if (imagerect.contains(p)) {
            double y1 = wave.v2;
            
            cv::Point2d dd(h.x*(this->xarray[(wave.dir+7)%8] - this->xarray[wave.dir]), h.y*(this->yarray[(wave.dir+7)%8] - this->yarray[wave.dir]));

            if (forced && cv::norm(dn - dp - dd) - cv::norm(h) > TAU ) {
                res2 = y0 + cv::norm(dn - dp)*(cv::norm(f0) + cv::norm(fn))/2.0;
            }
            else {
            
                cv::Point2d f1(timageP(p, 1), timageP(p, 0));
                if (std::isinf(cv::norm(f1)) || std::isnan(cv::norm(f1)))
                    f1 = fn;

                double A = -dd.y, B = dd.x, C = dd.y*dp.x - dd.x*dp.y;
                double den = A*fn.x + B*fn.y;
                double t = (A*dn.x + B*dn.y + C)/den, epsilon;

                cv::Point2d x(dn.x - t*fn.x, dn.y - t*fn.y); 

                if (fabs(dd.x) > 0.0 && fabs(den) > 0.0)
                    epsilon = (x.x - dp.x)/dd.x;
                else if (fabs(dd.y) > 0.0 && fabs(den) > 0.0)
                    epsilon = (x.y - dp.y)/dd.y;
                else if (fabs(den) == 0.0 && cv::norm(dd) > 0.0) {
                    double dist = fabs(A*dn.x + B*dn.y + C)/sqrt(A*A + B*B);
                    epsilon = (cv::norm(dn - dp) - dist)/(fabs(dd.x) + fabs(dd.y)); 
                }
                else
                    epsilon = 0.0;

                if (epsilon < 0.0)
                    epsilon = 0.0;
                else if (epsilon > 1.0)
                    epsilon = 1.0;

                cv::Point2d wp1 = dp + epsilon*dd;
                double t_2 = epsilon*epsilon;
                double t_3 = epsilon*t_2;

                double ft = (2.0*t_3 - 3.0*t_2 + 1.0)*cv::norm(f0) +
                        (t_3 - 2.0*t_2 + epsilon)*wave.fm2 +
                        (-2.0*t_3 + 3.0*t_2)*cv::norm(f1) +
                        (t_3 - t_2)*wave.fm3;

                res2 = (2.0*t_3 - 3.0*t_2 + 1.0)*y0 +
                        (t_3 - 2.0*t_2 + epsilon)*wave.m2 +
                        (-2.0*t_3 + 3.0*t_2)*y1 +
                        (t_3 - t_2)*wave.m3 +
                        cv::norm(dn - wp1)*(ft+cv::norm(fn))/2.0;
            }
            
        }
        
        val = std::min(res1, res2);
        
    }
    
    return val;
    
}



double TurboWMMPchip::HopfLax(cv::Mat &image, cv::Mat &u_surface, TIsoWavefrontP &wave, cv::Point &neigh, cv::Point2d &h, bool forced) {
    
    cv::Point2d f0(timageP(wave.p, 1), timageP(wave.p, 0)), fn(timageP(neigh, 1), timageP(neigh, 0));
    double y0 = wave.v0;
    
    if (std::isinf(cv::norm(f0)) || std::isnan(cv::norm(f0)))
        f0 = fn;
    
    double val = MAX_VAL;
    if (wave.dir == -1) {
        cv::Point2d diff(h.x*(neigh.x - wave.p.x), h.y*(neigh.y - wave.p.y));
        val = y0 + cv::norm(diff)*(cv::norm(f0) + cv::norm(fn))/2.0;
    }
    else {
        cv::Rect imagerect(0, 0, image.cols, image.rows);
        
        cv::Point p(wave.p.x + this->xarray[(wave.dir+1)%8] - this->xarray[wave.dir], 
                wave.p.y + this->yarray[(wave.dir+1)%8] - this->yarray[wave.dir]);
        double res1 = MAX_VAL;
        cv::Point2d dp(h.x*wave.p.x, h.y*wave.p.y), dn(h.x*neigh.x, h.y*neigh.y);
        
        if (imagerect.contains(p)) {
            double y1 = wave.v1;
            
            cv::Point2d dd(h.x*(this->xarray[(wave.dir+1)%8] - this->xarray[wave.dir]), h.y*(this->yarray[(wave.dir+1)%8] - this->yarray[wave.dir]));
            
            if (forced && cv::norm(dn - dp - dd) - cv::norm(h) > TAU ) {
                res1 = y0 + cv::norm(dn - dp)*(cv::norm(f0) + cv::norm(fn))/2.0;
            }
            else {
            
                cv::Point2d f1(timageP(p, 1), timageP(p, 0));
                if (std::isinf(cv::norm(f1)) || std::isnan(cv::norm(f1)))
                    f1 = fn;

                if (cv::norm(dn - dp - dd) < TAU) {
                    double res = y0 + cv::norm(dd)*(cv::norm(fn) + cv::norm(f0))/2.0;
                    res1 = std::min(y1, res);
                }
                else if (cv::norm(dn - dp) < TAU) {
                    double res = y1 + cv::norm(dd)*(cv::norm(fn) + cv::norm(f1))/2.0;
                    res1 = std::min(y0, res);
                }
                else {
                    cv::Point2d xy = dn - dp;
                    double nxy = cv::norm(xy);
                    double nyz = cv::norm(dd);

                    double c_alpha = (xy.x*dd.x + xy.y*dd.y)/(nxy*nyz);
                    double c_delta = (y1 - y0)/nyz;

                    if (nyz == 0.0 || c_alpha <= c_delta || c_alpha == 1.0) {
                        res1 = y0 + nxy*(cv::norm(fn) + cv::norm(f0))/2.0;
                    }
                    else {
                        cv::Point2d xz = dn - dp - dd;
                        double nxz = cv::norm(xz);
                        double c_beta = (xz.x*dd.x + xz.y*dd.y)/(nxz*nyz);

                        if (c_delta <= c_beta) {
                            res1 = y1 + nxz*(cv::norm(fn) + cv::norm(f1))/2.0;
                        }
                        else {
                            double s_delta = sqrt(1.0 - c_delta*c_delta);
                            double dist = (c_alpha*c_delta + sqrt(1.0 - c_alpha*c_alpha)*s_delta)*nxy;
                            double yzdist = sqrt(nxy*nxy - dist*dist);
                            double t = yzdist/(s_delta*nyz);

                            cv::Point2d respos = dp + t*dd;
                            double t_2 = t*t;
                            double t_3 = t*t_2;

                            double ft = (2.0*t_3 - 3.0*t_2 + 1.0)*cv::norm(f0) +
                                    (t_3 - 2.0*t_2 + t)*wave.fm0 +
                                    (-2.0*t_3 + 3.0*t_2)*cv::norm(f1) +
                                    (t_3 - t_2)*wave.fm1;

                            res1 = (2.0*t_3 - 3.0*t_2 + 1.0)*y0 +
                                    (t_3 - 2.0*t_2 + t)*wave.m0 +
                                    (-2.0*t_3 + 3.0*t_2)*y1 +
                                    (t_3 - t_2)*wave.m1 +
                                    cv::norm(dn - respos)*(ft+cv::norm(fn))/2.0;
                        }
                    }
                }
            }
            
        }
        
        p = cv::Point(wave.p.x + this->xarray[(wave.dir+7)%8] - this->xarray[wave.dir], 
                wave.p.y + this->yarray[(wave.dir+7)%8] - this->yarray[wave.dir]);
        double res2 = MAX_VAL;
        
        if (imagerect.contains(p)) {
            double y1 = wave.v2;
            
            cv::Point2d dd(h.x*(this->xarray[(wave.dir+7)%8] - this->xarray[wave.dir]), h.y*(this->yarray[(wave.dir+7)%8] - this->yarray[wave.dir]));

            if (forced && cv::norm(dn - dp - dd) - cv::norm(h) > TAU ) {
                res2 = y0 + cv::norm(dn - dp)*(cv::norm(f0) + cv::norm(fn))/2.0;
            }
            else {
            
                cv::Point2d f1(timageP(p, 1), timageP(p, 0));
                if (std::isinf(cv::norm(f1)) || std::isnan(cv::norm(f1)))
                    f1 = fn;

                if (cv::norm(dn - dp - dd) < TAU) {
                    double res = y0 + cv::norm(dd)*(cv::norm(fn) + cv::norm(f0))/2.0;
                    res2 = std::min(y1, res);
                }
                else if (cv::norm(dn - dp) < TAU) {
                    double res = y1 + cv::norm(dd)*(cv::norm(fn) + cv::norm(f1))/2.0;
                    res2 = std::min(y0, res);
                }
                else {

                    cv::Point2d xy = dn - dp;
                    double nxy = cv::norm(xy);
                    double nyz = cv::norm(dd);

                    double c_alpha = (xy.x*dd.x + xy.y*dd.y)/(nxy*nyz);
                    double c_delta = (y1 - y0)/nyz;

                    if (nyz == 0.0 || c_alpha <= c_delta || c_alpha == 1.0) {
                        res2 = y0 + nxy*(cv::norm(fn) + cv::norm(f0))/2.0;
                    }
                    else {
                        cv::Point2d xz = dn - dp - dd;
                        double nxz = cv::norm(xz);
                        double c_beta = (xz.x*dd.x + xz.y*dd.y)/(nxz*nyz);

                        if (c_delta <= c_beta) {
                            res2 = y1 + nxz*(cv::norm(fn) + cv::norm(f1))/2.0;
                        }
                        else {
                            double s_delta = sqrt(1.0 - c_delta*c_delta);
                            double dist = (c_alpha*c_delta + sqrt(1.0 - c_alpha*c_alpha)*s_delta)*nxy;
                            double yzdist = sqrt(nxy*nxy - dist*dist);
                            double t = yzdist/(s_delta*nyz);
                            cv::Point2d respos = dp + t*dd;
                            double t_2 = t*t;
                            double t_3 = t*t_2;

                            double ft = (2.0*t_3 - 3.0*t_2 + 1.0)*cv::norm(f0) +
                                    (t_3 - 2.0*t_2 + t)*wave.fm2 +
                                    (-2.0*t_3 + 3.0*t_2)*cv::norm(f1) +
                                    (t_3 - t_2)*wave.fm3;

                            res2 = (2.0*t_3 - 3.0*t_2 + 1.0)*y0 +
                                    (t_3 - 2.0*t_2 + t)*wave.m2 +
                                    (-2.0*t_3 + 3.0*t_2)*y1 +
                                    (t_3 - t_2)*wave.m3 +
                                    cv::norm(dn - respos)*(ft+cv::norm(fn))/2.0;
                        }
                    }
                }
            }
        }
        val = std::min(res1, res2);
        
    }
    
    return val;
    
}




double TurboWMMPchip::GoldenSearch(cv::Mat &image, cv::Mat &u_surface, TIsoWavefrontP &wave, cv::Point &neigh, cv::Point2d &h, bool forced) {
    
    cv::Point2d f0(timageP(wave.p, 1), timageP(wave.p, 0)), fn(timageP(neigh, 1), timageP(neigh, 0));
    double y0 = wave.v0;
    
    if (std::isinf(cv::norm(f0)) || std::isnan(cv::norm(f0)))
        f0 = fn;
    
    double val = MAX_VAL;
    if (wave.dir == -1) {
        cv::Point2d diff(h.x*(neigh.x - wave.p.x), h.y*(neigh.y - wave.p.y));
        val = y0 + cv::norm(diff)*(cv::norm(f0) + cv::norm(fn))/2.0;
    }
    else {
        cv::Rect imagerect(0, 0, image.cols, image.rows);
        
        cv::Point p(wave.p.x + this->xarray[(wave.dir+1)%8] - this->xarray[wave.dir], 
                wave.p.y + this->yarray[(wave.dir+1)%8] - this->yarray[wave.dir]);
        double res1 = MAX_VAL;
        cv::Point2d dp(h.x*wave.p.x, h.y*wave.p.y), dn(h.x*neigh.x, h.y*neigh.y);
        
        if (imagerect.contains(p)) {
            double y1 = wave.v1;
            
            cv::Point2d dd(h.x*(this->xarray[(wave.dir+1)%8] - this->xarray[wave.dir]), h.y*(this->yarray[(wave.dir+1)%8] - this->yarray[wave.dir]));
            
            if (forced && cv::norm(dn - dp - dd) - cv::norm(h) > TAU ) {
                res1 = y0 + cv::norm(dn - dp)*(cv::norm(f0) + cv::norm(fn))/2.0;
            }
            else {
            
                cv::Point2d f1(timageP(p, 1), timageP(p, 0));
                if (std::isinf(cv::norm(f1)) || std::isnan(cv::norm(f1)))
                    f1 = fn;

                double a = 0.0, b = 1.0, x1 = a + (1-RESPHI)*(b - a), x2 = a + RESPHI*(b - a),
                f_x1 = MAX_VAL, f_x2 = MAX_VAL, i1, i2;

                cv::Point2d xtreme = dp + dd;

                cv::Point2d F_x1, F_x2;

                double f_a = y0 + cv::norm(dn - dp)*(cv::norm(f0) + cv::norm(fn))/2.0;
                double f_b = y1 + cv::norm(dn - xtreme)*(cv::norm(f1) + cv::norm(fn))/2.0;

                res1 = (f_a < f_b) ? f_a : f_b;

                F_x1 = (1.0 - x1)*f0 + x1*f1;
                F_x2 = (1.0 - x2)*f0 + x2*f1;

                double x1_2 = x1*x1, x1_3 = x1_2*x1;
                i1 = (2.0*x1_3 - 3.0*x1_2 + 1.0)*cv::norm(f0) +
                        (x1_3 - 2.0*x1_2 + x1)*wave.fm0 +
                        (-2.0*x1_3 + 3.0*x1_2)*cv::norm(f1) +
                        (x1_3 - x1_2)*wave.fm1;
                f_x1 = (2.0*x1_3 - 3.0*x1_2 + 1.0)*y0 +
                        (x1_3 - 2.0*x1_2 + x1)*wave.m0 +
                        (-2.0*x1_3 + 3.0*x1_2)*y1 +
                        (x1_3 - x1_2)*wave.m1 + cv::norm(dn - (1.0 - x1)*dp - x1*xtreme)*(i1 + cv::norm(fn))/2.0;
                double x2_2 = x2*x2, x2_3 = x2_2*x2;
                i2 = (2.0*x2_3 - 3.0*x2_2 + 1.0)*cv::norm(f0) +
                        (x2_3 - 2.0*x2_2 + x2)*wave.fm0 +
                        (-2.0*x2_3 + 3.0*x2_2)*cv::norm(f1) +
                        (x2_3 - x2_2)*wave.fm1;
                f_x2 = (2.0*x2_3 - 3.0*x2_2 + 1.0)*y0 +
                        (x2_3 - 2.0*x2_2 + x2)*wave.m0 +
                        (-2.0*x2_3 + 3.0*x2_2)*y1 +
                        (x2_3 - x2_2)*wave.m1 + cv::norm(dn - (1.0 - x2)*dp - x2*xtreme)*(i2 + cv::norm(fn))/2.0;

                while (fabs(b - a) > TAU) {
                    if(f_x1 < f_x2) {
                        b = x2; x2 = x1; f_x2 = f_x1; x1 = a + (1 - RESPHI)*(b - a);
                        F_x2 = F_x1;
                        F_x1 = (1.0 - x1)*f0 + x1*f1;
                        x1_2 = x1*x1; x1_3 = x1_2*x1;
                        i1 = (2.0*x1_3 - 3.0*x1_2 + 1.0)*cv::norm(f0) +
                                (x1_3 - 2.0*x1_2 + x1)*wave.fm0 +
                                (-2.0*x1_3 + 3.0*x1_2)*cv::norm(f1) +
                                (x1_3 - x1_2)*wave.fm1;
                        f_x1 = (2.0*x1_3 - 3.0*x1_2 + 1.0)*y0 +
                                (x1_3 - 2.0*x1_2 + x1)*wave.m0 +
                                (-2.0*x1_3 + 3.0*x1_2)*y1 +
                                (x1_3 - x1_2)*wave.m1 + cv::norm(dn - (1.0 - x1)*dp - x1*xtreme)*(i1 + cv::norm(fn))/2.0;
                    }
                    else {
                        a = x1; x1 = x2; f_x1 = f_x2; x2 = a + RESPHI*(b - a);
                        F_x1 = F_x2;
                        F_x2 = (1.0 - x2)*f0 + x2*f1;
                        x2_2 = x2*x2; x2_3 = x2_2*x2;
                        i2 = (2.0*x2_3 - 3.0*x2_2 + 1.0)*cv::norm(f0) +
                                (x2_3 - 2.0*x2_2 + x2)*wave.fm0 +
                                (-2.0*x2_3 + 3.0*x2_2)*cv::norm(f1) +
                                (x2_3 - x2_2)*wave.fm1;
                        f_x2 = (2.0*x2_3 - 3.0*x2_2 + 1.0)*y0 +
                                (x2_3 - 2.0*x2_2 + x2)*wave.m0 +
                                (-2.0*x2_3 + 3.0*x2_2)*y1 +
                                (x2_3 - x2_2)*wave.m1 + cv::norm(dn - (1.0 - x2)*dp - x2*xtreme)*(i2 + cv::norm(fn))/2.0;
                    }
                }

                res1 = std::min(res1, std::min(f_x1, f_x2));
            }
            
        }
        
        p = cv::Point(wave.p.x + this->xarray[(wave.dir+7)%8] - this->xarray[wave.dir], 
                wave.p.y + this->yarray[(wave.dir+7)%8] - this->yarray[wave.dir]);
        double res2 = MAX_VAL;
        
        if (imagerect.contains(p)) {
            double y1 = wave.v2;
            
            cv::Point2d dd(h.x*(this->xarray[(wave.dir+7)%8] - this->xarray[wave.dir]), h.y*(this->yarray[(wave.dir+7)%8] - this->yarray[wave.dir]));

            if (forced && cv::norm(dn - dp - dd) - cv::norm(h) > TAU ) {
                res2 = y0 + cv::norm(dn - dp)*(cv::norm(f0) + cv::norm(fn))/2.0;
            }
            else {
            
                cv::Point2d f1(timageP(p, 1), timageP(p, 0));
                if (std::isinf(cv::norm(f1)) || std::isnan(cv::norm(f1)))
                    f1 = fn;

                double a = 0.0, b = 1.0, x1 = a + (1-RESPHI)*(b - a), x2 = a + RESPHI*(b - a),
                f_x1 = MAX_VAL, f_x2 = MAX_VAL, i1, i2;

                cv::Point2d xtreme = dp + dd;

                cv::Point2d F_x1, F_x2;

                double f_a = y0 + cv::norm(dn - dp)*(cv::norm(f0) + cv::norm(fn))/2.0;
                double f_b = y1 + cv::norm(dn - xtreme)*(cv::norm(f1) + cv::norm(fn))/2.0;

                res2 = (f_a < f_b) ? f_a : f_b;

                F_x1 = (1.0 - x1)*f0 + x1*f1;
                F_x2 = (1.0 - x2)*f0 + x2*f1;

                double x1_2 = x1*x1, x1_3 = x1_2*x1;
                i1 = (2.0*x1_3 - 3.0*x1_2 + 1.0)*cv::norm(f0) +
                        (x1_3 - 2.0*x1_2 + x1)*wave.fm2 +
                        (-2.0*x1_3 + 3.0*x1_2)*cv::norm(f1) +
                        (x1_3 - x1_2)*wave.fm3;
                f_x1 = (2.0*x1_3 - 3.0*x1_2 + 1.0)*y0 +
                        (x1_3 - 2.0*x1_2 + x1)*wave.m2 +
                        (-2.0*x1_3 + 3.0*x1_2)*y1 +
                        (x1_3 - x1_2)*wave.m3 + cv::norm(dn - (1.0 - x1)*dp - x1*xtreme)*(i1 + cv::norm(fn))/2.0;
                double x2_2 = x2*x2, x2_3 = x2_2*x2;
                i2 = (2.0*x2_3 - 3.0*x2_2 + 1.0)*cv::norm(f0) +
                        (x2_3 - 2.0*x2_2 + x2)*wave.fm2 +
                        (-2.0*x2_3 + 3.0*x2_2)*cv::norm(f1) +
                        (x2_3 - x2_2)*wave.fm3;
                f_x2 = (2.0*x2_3 - 3.0*x2_2 + 1.0)*y0 +
                        (x2_3 - 2.0*x2_2 + x2)*wave.m2 +
                        (-2.0*x2_3 + 3.0*x2_2)*y1 +
                        (x2_3 - x2_2)*wave.m3 + cv::norm(dn - (1.0 - x2)*dp - x2*xtreme)*(i2 + cv::norm(fn))/2.0;

                while (fabs(b - a) > TAU) {
                    if(f_x1 < f_x2) {
                        b = x2; x2 = x1; f_x2 = f_x1; x1 = a + (1 - RESPHI)*(b - a);
                        F_x2 = F_x1;
                        F_x1 = (1.0 - x1)*f0 + x1*f1;
                        x1_2 = x1*x1; x1_3 = x1_2*x1;
                        i1 = (2.0*x1_3 - 3.0*x1_2 + 1.0)*cv::norm(f0) +
                                (x1_3 - 2.0*x1_2 + x1)*wave.fm2 +
                                (-2.0*x1_3 + 3.0*x1_2)*cv::norm(f1) +
                                (x1_3 - x1_2)*wave.fm3;
                        f_x1 = (2.0*x1_3 - 3.0*x1_2 + 1.0)*y0 +
                                (x1_3 - 2.0*x1_2 + x1)*wave.m2 +
                                (-2.0*x1_3 + 3.0*x1_2)*y1 +
                                (x1_3 - x1_2)*wave.m3 + cv::norm(dn - (1.0 - x1)*dp - x1*xtreme)*(i1 + cv::norm(fn))/2.0;
                    }
                    else {
                        a = x1; x1 = x2; f_x1 = f_x2; x2 = a + RESPHI*(b - a);
                        F_x1 = F_x2;
                        F_x2 = (1.0 - x2)*f0 + x2*f1;
                        x2_2 = x2*x2; x2_3 = x2_2*x2;
                        i2 = (2.0*x2_3 - 3.0*x2_2 + 1.0)*cv::norm(f0) +
                                (x2_3 - 2.0*x2_2 + x2)*wave.fm2 +
                                (-2.0*x2_3 + 3.0*x2_2)*cv::norm(f1) +
                                (x2_3 - x2_2)*wave.fm3;
                        f_x2 = (2.0*x2_3 - 3.0*x2_2 + 1.0)*y0 +
                                (x2_3 - 2.0*x2_2 + x2)*wave.m2 +
                                (-2.0*x2_3 + 3.0*x2_2)*y1 +
                                (x2_3 - x2_2)*wave.m3 + cv::norm(dn - (1.0 - x2)*dp - x2*xtreme)*(i2 + cv::norm(fn))/2.0;
                    }
                }
                res2 = std::min(res2, std::min(f_x1, f_x2));
            }
        }
        val = std::min(res1, res2);
        
    }
    
    return val;
    
}



void TurboWMMPchip::getPCHIP(double *y, std::pair<double, double> *m, int pos) {
    
    double d0 = y[(pos+1)%8] - y[pos], d1 = y[(pos+2)%8] - y[(pos+1)%8];

    m[pos].first = (3.0*d0 - d1)/2.0;
    if ((m[pos].first * d0) <= 0.0)
        m[pos].first = 0.0;
    else if (((d1 * d0) <= 0.0) && (fabs(m[pos].first) > fabs(3.0*d0)))
        m[pos].first = 3.0*d0;
    
    m[pos].second = (d0*d1 <= 0.0) ? 0.0 : 2.0*d0*d1/(d0 + d1);
    m[(pos+1)%8].first = m[pos].second;

    m[(pos+1)%8].second = (3.0*d1 - d0)/2.0;
    if ((m[(pos+1)%8].second * d1) <= 0.0)
        m[(pos+1)%8].second = 0.0;
    else if (((d0 * d1) <= 0.0) && (fabs(m[(pos+1)%8].second) > fabs(3.0*d1)))
        m[(pos+1)%8].second = 3.0*d1;

}



TurboWMMPchip::~TurboWMMPchip() {
    
}

