#include "../header/TurboWMMLinear.h"
#include <opencv/cv.h>
#include <map>

#define MAX_VAL 100000
#define P_FAR   0
#define P_ALIVE 1
#define P_TRIAL 2
#define RESPHI  (sqrt(5.0)-1.0)/2.0
#define TAU     1e-03

#define timageL(p,k) reinterpret_cast<double *>(image.data)[(int) (p.y + image.rows*(p.x + image.cols*k))]
#define tdistanceL(p) reinterpret_cast<double *>(u_surface.data)[(int) (p.y + u_surface.rows*p.x)]

TurboWMMLinear::TurboWMMLinear() {
    this->yarray[0] = -1; this->yarray[1] = -1; this->yarray[2] = -1; this->yarray[3] = 0;
    this->yarray[4] = 1; this->yarray[5] = 1; this->yarray[6] = 1; this->yarray[7] = 0;
    this->xarray[0] = -1; this->xarray[1] = 0; this->xarray[2] = 1; this->xarray[3] = 1;
    this->xarray[4] = 1; this->xarray[5] = 0; this->xarray[6] = -1; this->xarray[7] = -1;
}

cv::Mat TurboWMMLinear::AniSurfaceGrad(cv::Mat& image, cv::vector<cv::Point>& initials, cv::Point2d &h) {

    cv::Mat u_surface = MAX_VAL*cv::Mat::ones(image.rows, image.cols, CV_64FC1);
    cv::Mat state = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
    
    std::multimap<double, TIsoWavefrontL > trial_set;
    std::map<int, std::multimap<double, TIsoWavefrontL >::iterator> mapa_trial;
    
    std::multimap<double, TIsoWavefrontL >::iterator trial_set_it;
    std::map<int, std::multimap<double, TIsoWavefrontL >::iterator>::iterator mapa_trial_it;
    std::pair<double, TIsoWavefrontL > pr_trial;
    std::pair<int, std::multimap<double, TIsoWavefrontL >::iterator> pr_mapa;
    int key, i;
    TIsoWavefrontL  winner, new_w;
            
    cv::Rect imagerect(0, 0, image.cols, image.rows);
    
    // Initialization
    for (i = 0; i < (int) initials.size(); i++) {
        key = initials[i].y*u_surface.cols + initials[i].x;
        if (mapa_trial.find(key) == mapa_trial.end()) {
            tdistanceL(initials[i]) = 0.0;
            winner.dir = -1;
            winner.v0 = 0.0;
            winner.p = initials[i];
            state.at<unsigned char>(initials[i]) = P_TRIAL;
            pr_trial = std::pair<double, TIsoWavefrontL >(0.0, winner);
            trial_set_it = trial_set.insert(pr_trial);
            pr_mapa = std::pair<int, std::multimap<double, TIsoWavefrontL >::iterator>(key, trial_set_it);
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
            this->isnewpos[i] = false;
            this->valcenter[i] = MAX_VAL;
            cv::Point neigh(winner.p.x + this->xarray[i], winner.p.y + this->yarray[i]);
            if (imagerect.contains(neigh) && state.at<unsigned char>(neigh) != P_ALIVE) {
                double val_neigh = this->Gradient(image, u_surface, winner, neigh, h);
                this->valcenter[i] = tdistanceL(neigh);
                if (val_neigh < this->valcenter[i]) {
                    this->isnewpos[i] = true;
                    this->valcenter[i] = val_neigh;
                }
            }
        }
        
        for (int i=0; i < 8; i++) {
            if (this->isnewpos[i]) {
                new_w.p = cv::Point(winner.p.x + this->xarray[i], winner.p.y + this->yarray[i]);
                if (this->valcenter[i] <= winner.v0  && this->valcenter[i] <= this->valcenter[(i+1)%8] && this->valcenter[i] <= this->valcenter[(i+7)%8] && 
                    (i%2 == 0 || (this->valcenter[i] <= this->valcenter[(i+2)%8] && this->valcenter[i] <= this->valcenter[(i+6)%8]))) {
                    double val_neigh = this->Gradient(image, u_surface, winner, new_w.p, h, true);
                    this->valcenter[i] = tdistanceL(new_w.p);
                    if (val_neigh < this->valcenter[i]) {
                        this->isnewpos[i] = true;
                        this->valcenter[i] = val_neigh;
                    }
                }
            }
        }
        
        
        for (int i=0; i < 8; i++) {
            if (this->isnewpos[i]) {
                new_w.p = cv::Point(winner.p.x + this->xarray[i], winner.p.y + this->yarray[i]);
                new_w.dir = i;
                key = new_w.p.y*u_surface.cols + new_w.p.x;
                if (state.at<unsigned char>(new_w.p) == P_TRIAL) {
                    mapa_trial_it = mapa_trial.find(key);
                    trial_set.erase(mapa_trial_it->second);
                    mapa_trial.erase(mapa_trial_it);
                }
                else {
                    state.at<unsigned char>(new_w.p) = P_TRIAL;
                }
                new_w.v0 = valcenter[i];
                new_w.v1 = valcenter[(i+1)%8];
                new_w.v2 = valcenter[(i+7)%8];
                pr_trial = std::pair<double, TIsoWavefrontL >(new_w.v0, new_w);
                trial_set_it = trial_set.insert(pr_trial);
                pr_mapa = std::pair<int, std::multimap<double, TIsoWavefrontL >::iterator>(key, trial_set_it);
                mapa_trial.insert(pr_mapa);
                tdistanceL(new_w.p) = new_w.v0;
            }
        }
        
        
    }
    
    return u_surface;
    
}

cv::Mat TurboWMMLinear::AniSurfaceHL(cv::Mat& image, cv::vector<cv::Point>& initials, cv::Point2d &h) {

    cv::Mat u_surface = MAX_VAL*cv::Mat::ones(image.rows, image.cols, CV_64FC1);
    cv::Mat state = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
    
    std::multimap<double, TIsoWavefrontL > trial_set;
    std::map<int, std::multimap<double, TIsoWavefrontL >::iterator> mapa_trial;
    
    std::multimap<double, TIsoWavefrontL >::iterator trial_set_it;
    std::map<int, std::multimap<double, TIsoWavefrontL >::iterator>::iterator mapa_trial_it;
    std::pair<double, TIsoWavefrontL > pr_trial;
    std::pair<int, std::multimap<double, TIsoWavefrontL >::iterator> pr_mapa;
    int key, i;
    TIsoWavefrontL  winner, new_w;
            
    cv::Rect imagerect(0, 0, image.cols, image.rows);
    
    // Initialization
    for (i = 0; i < (int) initials.size(); i++) {
        key = initials[i].y*u_surface.cols + initials[i].x;
        if (mapa_trial.find(key) == mapa_trial.end()) {
            tdistanceL(initials[i]) = 0.0;
            winner.dir = -1;
            winner.v0 = 0.0;
            winner.p = initials[i];
            state.at<unsigned char>(initials[i]) = P_TRIAL;
            pr_trial = std::pair<double, TIsoWavefrontL >(0.0, winner);
            trial_set_it = trial_set.insert(pr_trial);
            pr_mapa = std::pair<int, std::multimap<double, TIsoWavefrontL >::iterator>(key, trial_set_it);
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
            this->isnewpos[i] = false;
            this->valcenter[i] = MAX_VAL;
            cv::Point neigh(winner.p.x + this->xarray[i], winner.p.y + this->yarray[i]);
            if (imagerect.contains(neigh) && state.at<unsigned char>(neigh) != P_ALIVE) {
                double val_neigh = this->HopfLax(image, u_surface, winner, neigh, h);
                this->valcenter[i] = tdistanceL(neigh);
                if (val_neigh < this->valcenter[i]) {
                    this->isnewpos[i] = true;
                    this->valcenter[i] = val_neigh;
                }
            }
        }
        
        for (int i=0; i < 8; i++) {
            if (this->isnewpos[i]) {
                new_w.p = cv::Point(winner.p.x + this->xarray[i], winner.p.y + this->yarray[i]);
                if (this->valcenter[i] <= winner.v0  && this->valcenter[i] <= this->valcenter[(i+1)%8] && this->valcenter[i] <= this->valcenter[(i+7)%8] && 
                    (i%2 == 0 || (this->valcenter[i] <= this->valcenter[(i+2)%8] && this->valcenter[i] <= this->valcenter[(i+6)%8]))) {
                    double val_neigh = this->HopfLax(image, u_surface, winner, new_w.p, h, true);
                    this->valcenter[i] = tdistanceL(new_w.p);
                    if (val_neigh < this->valcenter[i]) {
                        this->isnewpos[i] = true;
                        this->valcenter[i] = val_neigh;
                    }
                }
            }
        }
        
        
        for (int i=0; i < 8; i++) {
            if (this->isnewpos[i]) {
                new_w.p = cv::Point(winner.p.x + this->xarray[i], winner.p.y + this->yarray[i]);
                new_w.dir = i;
                key = new_w.p.y*u_surface.cols + new_w.p.x;
                if (state.at<unsigned char>(new_w.p) == P_TRIAL) {
                    mapa_trial_it = mapa_trial.find(key);
                    trial_set.erase(mapa_trial_it->second);
                    mapa_trial.erase(mapa_trial_it);
                }
                else {
                    state.at<unsigned char>(new_w.p) = P_TRIAL;
                }

                new_w.v0 = valcenter[i];
                new_w.v1 = valcenter[(i+1)%8];
                new_w.v2 = valcenter[(i+7)%8];
                pr_trial = std::pair<double, TIsoWavefrontL >(new_w.v0, new_w);
                trial_set_it = trial_set.insert(pr_trial);
                pr_mapa = std::pair<int, std::multimap<double, TIsoWavefrontL >::iterator>(key, trial_set_it);
                mapa_trial.insert(pr_mapa);
                tdistanceL(new_w.p) = new_w.v0;
            }
        }
        
        
    }
    
    return u_surface;
    
}


cv::Mat TurboWMMLinear::AniSurfaceGS(cv::Mat& image, cv::vector<cv::Point>& initials, cv::Point2d &h) {

    cv::Mat u_surface = MAX_VAL*cv::Mat::ones(image.rows, image.cols, CV_64FC1);
    cv::Mat state = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
    
    std::multimap<double, TIsoWavefrontL > trial_set;
    std::map<int, std::multimap<double, TIsoWavefrontL >::iterator> mapa_trial;
    
    std::multimap<double, TIsoWavefrontL >::iterator trial_set_it;
    std::map<int, std::multimap<double, TIsoWavefrontL >::iterator>::iterator mapa_trial_it;
    std::pair<double, TIsoWavefrontL > pr_trial;
    std::pair<int, std::multimap<double, TIsoWavefrontL >::iterator> pr_mapa;
    int key, i;
    TIsoWavefrontL  winner, new_w;
            
    cv::Rect imagerect(0, 0, image.cols, image.rows);
    
    // Initialization
    for (i = 0; i < (int) initials.size(); i++) {
        key = initials[i].y*u_surface.cols + initials[i].x;
        if (mapa_trial.find(key) == mapa_trial.end()) {
            tdistanceL(initials[i]) = 0.0;
            winner.dir = -1;
            winner.v0 = 0.0;
            winner.p = initials[i];
            state.at<unsigned char>(initials[i]) = P_TRIAL;
            pr_trial = std::pair<double, TIsoWavefrontL >(0.0, winner);
            trial_set_it = trial_set.insert(pr_trial);
            pr_mapa = std::pair<int, std::multimap<double, TIsoWavefrontL >::iterator>(key, trial_set_it);
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
            this->isnewpos[i] = false;
            this->valcenter[i] = MAX_VAL;
            cv::Point neigh(winner.p.x + this->xarray[i], winner.p.y + this->yarray[i]);
            if (imagerect.contains(neigh) && state.at<unsigned char>(neigh) != P_ALIVE) {
                double val_neigh = this->GoldenSearch(image, u_surface, winner, neigh, h);
                this->valcenter[i] = tdistanceL(neigh);
                if (val_neigh < this->valcenter[i]) {
                    this->isnewpos[i] = true;
                    this->valcenter[i] = val_neigh;
                }
            }
        }
        
        for (int i=0; i < 8; i++) {
            if (this->isnewpos[i]) {
                new_w.p = cv::Point(winner.p.x + this->xarray[i], winner.p.y + this->yarray[i]);
                if (this->valcenter[i] <= winner.v0  && this->valcenter[i] <= this->valcenter[(i+1)%8] && this->valcenter[i] <= this->valcenter[(i+7)%8] && 
                    (i%2 == 0 || (this->valcenter[i] <= this->valcenter[(i+2)%8] && this->valcenter[i] <= this->valcenter[(i+6)%8]))) {
                    double val_neigh = this->GoldenSearch(image, u_surface, winner, new_w.p, h, true);
                    this->valcenter[i] = tdistanceL(new_w.p);
                    if (val_neigh < this->valcenter[i]) {
                        this->isnewpos[i] = true;
                        this->valcenter[i] = val_neigh;
                    }
                }
            }
        }
        
        for (int i=0; i < 8; i++) {
            if (this->isnewpos[i]) {
                new_w.dir = i;
                new_w.p = cv::Point(winner.p.x + this->xarray[i], winner.p.y + this->yarray[i]);
                key = new_w.p.y*u_surface.cols + new_w.p.x;
                if (state.at<unsigned char>(new_w.p) == P_TRIAL) {
                    mapa_trial_it = mapa_trial.find(key);
                    trial_set.erase(mapa_trial_it->second);
                    mapa_trial.erase(mapa_trial_it);
                }
                else {
                    state.at<unsigned char>(new_w.p) = P_TRIAL;
                }
                new_w.v0 = valcenter[i];
                new_w.v1 = valcenter[(i+1)%8];
                new_w.v2 = valcenter[(i+7)%8];
                pr_trial = std::pair<double, TIsoWavefrontL >(new_w.v0, new_w);
                trial_set_it = trial_set.insert(pr_trial);
                pr_mapa = std::pair<int, std::multimap<double, TIsoWavefrontL >::iterator>(key, trial_set_it);
                mapa_trial.insert(pr_mapa);
                tdistanceL(new_w.p) = new_w.v0;
            }
        }
        
        
    }
    
    return u_surface;
    
}



double TurboWMMLinear::Gradient(cv::Mat &image, cv::Mat &u_surface, TIsoWavefrontL &wave, cv::Point &neigh, cv::Point2d &h, bool forced) {
    
    cv::Point2d f0(timageL(wave.p, 1), timageL(wave.p, 0)), fn(timageL(neigh, 1), timageL(neigh, 0));
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
                cv::Point2d f1(timageL(p, 1), timageL(p, 0));
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
                res1 = (1.0 - epsilon)*y0 + epsilon*y1 + cv::norm(dn - wp1)*((1.0 - epsilon)*cv::norm(f0)+epsilon*cv::norm(f1)+cv::norm(fn))/2.0;
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
            
                cv::Point2d f1(timageL(p, 1), timageL(p, 0));
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
                res2 = (1.0 - epsilon)*y0 + epsilon*y1 + cv::norm(dn - wp1)*((1.0 - epsilon)*cv::norm(f0)+epsilon*cv::norm(f1)+cv::norm(fn))/2.0;
            }
        }
        
        val = std::min(res1, res2);
        
        
    }
    
    return val;
    
}



double TurboWMMLinear::HopfLax(cv::Mat &image, cv::Mat &u_surface, TIsoWavefrontL &wave, cv::Point &neigh, cv::Point2d &h, bool forced) {
    
    cv::Point2d f0(timageL(wave.p, 1), timageL(wave.p, 0)), fn(timageL(neigh, 1), timageL(neigh, 0));
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
            
                cv::Point2d f1(timageL(p, 1), timageL(p, 0));
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
                            res1 = (1.0 - t)*y0 + t*y1 + cv::norm(dn - respos)*(cv::norm(fn) + (1.0 - t)*cv::norm(f0) + t*cv::norm(f1))/2.0;
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
            
                cv::Point2d f1(timageL(p, 1), timageL(p, 0));
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
                            res2 = (1.0 - t)*y0 + t*y1 + cv::norm(dn - respos)*(cv::norm(fn) + (1.0 - t)*cv::norm(f0) + t*cv::norm(f1))/2.0;
                        }
                    }
                }
            }
        }

        val = std::min(res1, res2);
        
    }
    
    return val;
    
}



double TurboWMMLinear::GoldenSearch(cv::Mat &image, cv::Mat &u_surface, TIsoWavefrontL &wave, cv::Point &neigh, cv::Point2d &h, bool forced) {
    
    cv::Point2d f0(timageL(wave.p, 1), timageL(wave.p, 0)), fn(timageL(neigh, 1), timageL(neigh, 0));
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
            
                cv::Point2d f1(timageL(p, 1), timageL(p, 0));
                if (std::isinf(cv::norm(f1)) || std::isnan(cv::norm(f1)))
                    f1 = fn;

                double a = 0.0, b = 1.0, x1 = a + (1-RESPHI)*(b - a), x2 = a + RESPHI*(b - a),
                f_x1 = MAX_VAL, f_x2 = MAX_VAL;

                cv::Point2d xtreme = dp + dd;

                cv::Point2d F_x1, F_x2;

                double f_a = y0 + cv::norm(dn - dp)*(cv::norm(f0) + cv::norm(fn))/2.0;
                double f_b = y1 + cv::norm(dn - xtreme)*(cv::norm(f1) + cv::norm(fn))/2.0;

                res1 = (f_a < f_b) ? f_a : f_b;

                F_x1 = (1.0 - x1)*f0 + x1*f1;
                F_x2 = (1.0 - x2)*f0 + x2*f1;

                f_x1 = (1.0 - x1)*y0 + x1*y1 + cv::norm(dn - (1.0 - x1)*dp - x1*xtreme)*((1.0 - x1)*cv::norm(f0) + x1*cv::norm(f1) + cv::norm(fn))/2.0;
                f_x2 = (1.0 - x2)*y0 + x2*y1 + cv::norm(dn - (1.0 - x2)*dp - x2*xtreme)*((1.0 - x2)*cv::norm(f0) + x2*cv::norm(f1) + cv::norm(fn))/2.0;

                while (fabs(b - a) > TAU) {
                    if(f_x1 < f_x2) {
                        b = x2; x2 = x1; f_x2 = f_x1; x1 = a + (1 - RESPHI)*(b - a);
                        F_x2 = F_x1;
                        F_x1 = (1.0 - x1)*f0 + x1*f1;
                        f_x1 = (1.0 - x1)*y0 + x1*y1 + cv::norm(dn - (1.0 - x1)*dp - x1*xtreme)*((1.0 - x1)*cv::norm(f0) + x1*cv::norm(f1) + cv::norm(fn))/2.0;
                    }
                    else {
                        a = x1; x1 = x2; f_x1 = f_x2; x2 = a + RESPHI*(b - a);
                        F_x1 = F_x2;
                        F_x2 = (1.0 - x2)*f0 + x2*f1;
                        f_x2 = (1.0 - x2)*y0 + x2*y1 + cv::norm(dn - (1.0 - x2)*dp - x2*xtreme)*((1.0 - x2)*cv::norm(f0) + x2*cv::norm(f1) + cv::norm(fn))/2.0;
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
            
                cv::Point2d f1(timageL(p, 1), timageL(p, 0));
                if (std::isinf(cv::norm(f1)) || std::isnan(cv::norm(f1)))
                    f1 = fn;

                double a = 0.0, b = 1.0, x1 = a + (1-RESPHI)*(b - a), x2 = a + RESPHI*(b - a),
                f_x1 = MAX_VAL, f_x2 = MAX_VAL;

                cv::Point2d xtreme = dp + dd;

                cv::Point2d F_x1, F_x2;

                double f_a = y0 + cv::norm(dn - dp)*(cv::norm(f0) + cv::norm(fn))/2.0;
                double f_b = y1 + cv::norm(dn - xtreme)*(cv::norm(f1) + cv::norm(fn))/2.0;

                res2 = (f_a < f_b) ? f_a : f_b;

                F_x1 = (1.0 - x1)*f0 + x1*f1;
                F_x2 = (1.0 - x2)*f0 + x2*f1;


                f_x1 = (1.0 - x1)*y0 + x1*y1 + cv::norm(dn - (1.0 - x1)*dp - x1*xtreme)*((1.0 - x1)*cv::norm(f0) + x1*cv::norm(f1) + cv::norm(fn))/2.0;
                f_x2 = (1.0 - x2)*y0 + x2*y1 + cv::norm(dn - (1.0 - x2)*dp - x2*xtreme)*((1.0 - x2)*cv::norm(f0) + x2*cv::norm(f1) + cv::norm(fn))/2.0;

                while (fabs(b - a) > TAU) {
                    if(f_x1 < f_x2) {
                        b = x2; x2 = x1; f_x2 = f_x1; x1 = a + (1 - RESPHI)*(b - a);
                        F_x2 = F_x1;
                        F_x1 = (1.0 - x1)*f0 + x1*f1;
                        f_x1 = (1.0 - x1)*y0 + x1*y1 + cv::norm(dn - (1.0 - x1)*dp - x1*xtreme)*((1.0 - x1)*cv::norm(f0) + x1*cv::norm(f1) + cv::norm(fn))/2.0;
                    }
                    else {
                        a = x1; x1 = x2; f_x1 = f_x2; x2 = a + RESPHI*(b - a);
                        F_x1 = F_x2;
                        F_x2 = (1.0 - x2)*f0 + x2*f1;
                        f_x2 = (1.0 - x2)*y0 + x2*y1 + cv::norm(dn - (1.0 - x2)*dp - x2*xtreme)*((1.0 - x2)*cv::norm(f0) + x2*cv::norm(f1) + cv::norm(fn))/2.0;
                    }
                }

                res2 = std::min(res2, std::min(f_x1, f_x2));
            }
        }
        val = std::min(res1, res2);
        
    }
    
    return val;
    
}



TurboWMMLinear::~TurboWMMLinear() {
    
}

