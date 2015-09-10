#include "../header/TurboWMMInterp.h"
#include <opencv/cv.h>
#include <map>

#define MAX_VAL 100000
#define P_FAR   0
#define P_ALIVE 1
#define P_TRIAL 2
#define RESPHI  (sqrt(5.0)-1.0)/2.0
#define TAU     1e-03

#define timageS(p,k) reinterpret_cast<double *>(image.data)[(int) (p.y + image.rows*(p.x + image.cols*k))]
#define tdistanceS(p) reinterpret_cast<double *>(u_surface.data)[(int) (p.y + u_surface.rows*p.x)]

TurboWMMInterp::TurboWMMInterp() {
    this->yarray[0] = -1; this->yarray[1] = -1; this->yarray[2] = -1; this->yarray[3] = 0;
    this->yarray[4] = 1; this->yarray[5] = 1; this->yarray[6] = 1; this->yarray[7] = 0;
    this->xarray[0] = -1; this->xarray[1] = 0; this->xarray[2] = 1; this->xarray[3] = 1;
    this->xarray[4] = 1; this->xarray[5] = 0; this->xarray[6] = -1; this->xarray[7] = -1;
}

cv::Mat TurboWMMInterp::AniSurfaceGrad(cv::Mat& image, cv::vector<cv::Point>& initials, cv::Point2d &h) {

    cv::Mat u_surface = MAX_VAL*cv::Mat::ones(image.rows, image.cols, CV_64FC1);
    cv::Mat state = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
    
    std::multimap<double, TIsoWavefrontI> trial_set;
    std::map<int, std::multimap<double, TIsoWavefrontI>::iterator> mapa_trial;
    
    std::multimap<double, TIsoWavefrontI>::iterator trial_set_it;
    std::map<int, std::multimap<double, TIsoWavefrontI>::iterator>::iterator mapa_trial_it;
    std::pair<double, TIsoWavefrontI> pr_trial;
    std::pair<int, std::multimap<double, TIsoWavefrontI>::iterator> pr_mapa;
    int key, i;
    TIsoWavefrontI winner, new_w;
    cv::Point neigh;
            
    cv::Rect imagerect(0, 0, image.cols, image.rows);
    
    // Initialization
    for (i = 0; i < (int) initials.size(); i++) {
        key = initials[i].y*u_surface.cols + initials[i].x;
        if (mapa_trial.find(key) == mapa_trial.end() && imagerect.contains(initials[i])) {
            tdistanceS(initials[i]) = 0.0;
            winner.dir = -1;
            winner.v0 = 0.0;
            winner.p = initials[i];
            state.at<unsigned char>(initials[i]) = P_TRIAL;
            pr_trial = std::pair<double, TIsoWavefrontI >(0.0, winner);
            trial_set_it = trial_set.insert(pr_trial);
            pr_mapa = std::pair<int, std::multimap<double, TIsoWavefrontI >::iterator>(key, trial_set_it);
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
            this->valcenter[i] = imagerect.contains(neigh) ? tdistanceS(neigh) : MAX_VAL;
            this->imcenter[i] = imagerect.contains(neigh) ? cv::norm(cv::Point2d(timageS(neigh,0), timageS(neigh,1))) : MAX_VAL;
            if (imagerect.contains(neigh) && state.at<unsigned char>(neigh) != P_ALIVE) {
                double val_neigh = this->Gradient(image, u_surface, winner, neigh, h);
                if (this->valcenter[i] - val_neigh > TAU ) {
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
                    this->valcenter[i] = tdistanceS(new_w.p);
                    if (val_neigh < this->valcenter[i]) {
                        this->isnewpos[i] = true;
                        this->valcenter[i] = val_neigh;
                    }
                }
            }
        }
        
        // Pchips
        for (int i=0; i < 8; i++) {
            this->getInterp(this->valcenter, this->ms, i);
            this->getInterpInverse(this->valcenter, this->ms2, i);
            this->getInterp(this->imcenter, this->ms3, i);
            this->getInterpInverse(this->imcenter, this->ms4, i);
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
                new_w.m0 = this->ms[i].p0;
                new_w.m1 = this->ms[i].p1;
                new_w.m2 = this->ms[i].p2;
                new_w.m3 = this->ms2[i].p0;
                new_w.m4 = this->ms2[i].p1;
                new_w.m5 = this->ms2[i].p2;
                new_w.fm0 = this->ms3[i].p0;
                new_w.fm1 = this->ms3[i].p1;
                new_w.fm2 = this->ms3[i].p2;
                new_w.fm3 = this->ms4[i].p0;
                new_w.fm4 = this->ms4[i].p1;
                new_w.fm5 = this->ms4[i].p2;
                pr_trial = std::pair<double, TIsoWavefrontI>(valcenter[i], new_w);
                trial_set_it = trial_set.insert(pr_trial);
                pr_mapa = std::pair<int, std::multimap<double, TIsoWavefrontI>::iterator>(key, trial_set_it);
                mapa_trial.insert(pr_mapa);
                                
                tdistanceS(new_w.p) = valcenter[i];
            }
        }
        
    }
    
    return u_surface;
    
}


cv::Mat TurboWMMInterp::AniSurfaceHL(cv::Mat& image, cv::vector<cv::Point>& initials, cv::Point2d &h) {

    cv::Mat u_surface = MAX_VAL*cv::Mat::ones(image.rows, image.cols, CV_64FC1);
    cv::Mat state = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
    
    std::multimap<double, TIsoWavefrontI> trial_set;
    std::map<int, std::multimap<double, TIsoWavefrontI>::iterator> mapa_trial;
    
    std::multimap<double, TIsoWavefrontI>::iterator trial_set_it;
    std::map<int, std::multimap<double, TIsoWavefrontI>::iterator>::iterator mapa_trial_it;
    std::pair<double, TIsoWavefrontI> pr_trial;
    std::pair<int, std::multimap<double, TIsoWavefrontI>::iterator> pr_mapa;
    int key, i;
    TIsoWavefrontI winner, new_w;
    cv::Point neigh;
            
    cv::Rect imagerect(0, 0, image.cols, image.rows);
    
    // Initialization
    for (i = 0; i < (int) initials.size(); i++) {
        key = initials[i].y*u_surface.cols + initials[i].x;
        if (mapa_trial.find(key) == mapa_trial.end() && imagerect.contains(initials[i])) {
            tdistanceS(initials[i]) = 0.0;
            winner.dir = -1;
            winner.v0 = 0.0;
            winner.p = initials[i];
            state.at<unsigned char>(initials[i]) = P_TRIAL;
            pr_trial = std::pair<double, TIsoWavefrontI >(0.0, winner);
            trial_set_it = trial_set.insert(pr_trial);
            pr_mapa = std::pair<int, std::multimap<double, TIsoWavefrontI >::iterator>(key, trial_set_it);
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
        
        //std::cout << "---------GANADOR = " << winner.p << winner.v0 << ", " << winner.v1 << ", " << winner.v2 << std::endl;
        
        trial_set.erase(trial_set_it);
        mapa_trial.erase(mapa_trial_it);
        
        state.at<unsigned char>(winner.p) = P_ALIVE;

        for (int i=0; i < 8; i++) {
            neigh = winner.p + cv::Point(this->xarray[i], this->yarray[i]);
            isnewpos[i] = false;
            this->valcenter[i] = imagerect.contains(neigh) ? tdistanceS(neigh) : MAX_VAL;
            this->imcenter[i] = imagerect.contains(neigh) ? cv::norm(cv::Point2d(timageS(neigh,0), timageS(neigh,1))) : MAX_VAL;
            if (imagerect.contains(neigh) && state.at<unsigned char>(neigh) != P_ALIVE) {
                double val_neigh = this->HopfLax(image, u_surface, winner, neigh, h);
                if (this->valcenter[i] - val_neigh > TAU ) {
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
                    this->valcenter[i] = tdistanceS(new_w.p);
                    if (val_neigh < this->valcenter[i]) {
                        this->isnewpos[i] = true;
                        this->valcenter[i] = val_neigh;
                    }
                }
            }
        }
        
        
        // Pchips
        for (int i=0; i < 8; i++) {
            this->getInterp(this->valcenter, this->ms, i);
            this->getInterpInverse(this->valcenter, this->ms2, i);
            this->getInterp(this->imcenter, this->ms3, i);
            this->getInterpInverse(this->imcenter, this->ms4, i);
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
                new_w.m0 = this->ms[i].p0;
                new_w.m1 = this->ms[i].p1;
                new_w.m2 = this->ms[i].p2;
                new_w.m3 = this->ms2[i].p0;
                new_w.m4 = this->ms2[i].p1;
                new_w.m5 = this->ms2[i].p2;
                new_w.fm0 = this->ms3[i].p0;
                new_w.fm1 = this->ms3[i].p1;
                new_w.fm2 = this->ms3[i].p2;
                new_w.fm3 = this->ms4[i].p0;
                new_w.fm4 = this->ms4[i].p1;
                new_w.fm5 = this->ms4[i].p2;
                pr_trial = std::pair<double, TIsoWavefrontI>(valcenter[i], new_w);
                trial_set_it = trial_set.insert(pr_trial);
                pr_mapa = std::pair<int, std::multimap<double, TIsoWavefrontI>::iterator>(key, trial_set_it);
                mapa_trial.insert(pr_mapa);
                                
                tdistanceS(new_w.p) = valcenter[i];
                
                //std::cout << "---ACTUALIZANDO = " << new_w.p << new_w.v0 << ", " << new_w.v1 << ", " << new_w.v2 << std::endl;
            }
        }
        
    }
    
    return u_surface;
    
}


cv::Mat TurboWMMInterp::AniSurfaceGS(cv::Mat& image, cv::vector<cv::Point>& initials, cv::Point2d &h) {

    cv::Mat u_surface = MAX_VAL*cv::Mat::ones(image.rows, image.cols, CV_64FC1);
    cv::Mat state = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
    
    std::multimap<double, TIsoWavefrontI> trial_set;
    std::map<int, std::multimap<double, TIsoWavefrontI>::iterator> mapa_trial;
    
    std::multimap<double, TIsoWavefrontI>::iterator trial_set_it;
    std::map<int, std::multimap<double, TIsoWavefrontI>::iterator>::iterator mapa_trial_it;
    std::pair<double, TIsoWavefrontI> pr_trial;
    std::pair<int, std::multimap<double, TIsoWavefrontI>::iterator> pr_mapa;
    int key, i;
    TIsoWavefrontI winner, new_w;
    cv::Point neigh;
            
    cv::Rect imagerect(0, 0, image.cols, image.rows);
    
    // Initialization
    for (i = 0; i < (int) initials.size(); i++) {
        key = initials[i].y*u_surface.cols + initials[i].x;
        if (mapa_trial.find(key) == mapa_trial.end() && imagerect.contains(initials[i])) {
            tdistanceS(initials[i]) = 0.0;
            winner.dir = -1;
            winner.v0 = 0.0;
            winner.p = initials[i];
            state.at<unsigned char>(initials[i]) = P_TRIAL;
            pr_trial = std::pair<double, TIsoWavefrontI >(0.0, winner);
            trial_set_it = trial_set.insert(pr_trial);
            pr_mapa = std::pair<int, std::multimap<double, TIsoWavefrontI >::iterator>(key, trial_set_it);
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
            this->valcenter[i] = imagerect.contains(neigh) ? tdistanceS(neigh) : MAX_VAL;
            this->imcenter[i] = imagerect.contains(neigh) ? cv::norm(cv::Point2d(timageS(neigh,0), timageS(neigh,1))) : MAX_VAL;
            if (imagerect.contains(neigh) && state.at<unsigned char>(neigh) != P_ALIVE) {
                double val_neigh = this->GoldenSearch(image, u_surface, winner, neigh, h);
                if (this->valcenter[i] - val_neigh > TAU ) {
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
                    this->valcenter[i] = tdistanceS(new_w.p);
                    if (val_neigh < this->valcenter[i]) {
                        this->isnewpos[i] = true;
                        this->valcenter[i] = val_neigh;
                    }
                }
            }
        }
        
        
        // Pchips
        for (int i=0; i < 8; i++) {
            this->getInterp(this->valcenter, this->ms, i);
            this->getInterpInverse(this->valcenter, this->ms2, i);
            this->getInterp(this->imcenter, this->ms3, i);
            this->getInterpInverse(this->imcenter, this->ms4, i);
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
                new_w.m0 = this->ms[i].p0;
                new_w.m1 = this->ms[i].p1;
                new_w.m2 = this->ms[i].p2;
                new_w.m3 = this->ms2[i].p0;
                new_w.m4 = this->ms2[i].p1;
                new_w.m5 = this->ms2[i].p2;
                new_w.fm0 = this->ms3[i].p0;
                new_w.fm1 = this->ms3[i].p1;
                new_w.fm2 = this->ms3[i].p2;
                new_w.fm3 = this->ms4[i].p0;
                new_w.fm4 = this->ms4[i].p1;
                new_w.fm5 = this->ms4[i].p2;
                pr_trial = std::pair<double, TIsoWavefrontI>(valcenter[i], new_w);
                trial_set_it = trial_set.insert(pr_trial);
                pr_mapa = std::pair<int, std::multimap<double, TIsoWavefrontI>::iterator>(key, trial_set_it);
                mapa_trial.insert(pr_mapa);
                                
                tdistanceS(new_w.p) = valcenter[i];
            }
        }
        
    }
    
    return u_surface;
    
}


double TurboWMMInterp::Gradient(cv::Mat &image, cv::Mat &u_surface, TIsoWavefrontI &wave, cv::Point &neigh, cv::Point2d &h, bool forced) {
    
    cv::Point2d f0(timageS(wave.p, 1), timageS(wave.p, 0)), fn(timageS(neigh, 1), timageS(neigh, 0));
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
            
                cv::Point2d f1(timageS(p, 1), timageS(p, 0));
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

                double fs = wave.fm0*epsilon*epsilon + wave.fm1*epsilon + wave.fm2;

                res1 = wave.m0*epsilon*epsilon + wave.m1*epsilon + wave.m2 + cv::norm(dn - wp1)*(fs+cv::norm(fn))/2.0;

                if (res1 < y0) res1 = (1.0 - epsilon)*y0 + epsilon*y1 + cv::norm(dn - wp1)*((1.0 - epsilon)*cv::norm(f0) + epsilon*cv::norm(f1)+cv::norm(fn))/2.0;
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
            
                cv::Point2d f1(timageS(p, 1), timageS(p, 0));
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
                double fs = wave.fm3*epsilon*epsilon + wave.fm4*epsilon + wave.fm5;

                res2 = wave.m3*epsilon*epsilon + wave.m4*epsilon + wave.m5 + cv::norm(dn - wp1)*(fs+cv::norm(fn))/2.0;

                if (res2 < y0) res2 = (1.0 - epsilon)*y0 + epsilon*y1 + cv::norm(dn - wp1)*((1.0 - epsilon)*cv::norm(f0) + epsilon*cv::norm(f1)+cv::norm(fn))/2.0;
            }
            
        }
        
        val = std::min(res1, res2);
        
    }
    
    return val;
    
}


double TurboWMMInterp::HopfLax(cv::Mat &image, cv::Mat &u_surface, TIsoWavefrontI &wave, cv::Point &neigh, cv::Point2d &h, bool forced) {
    
    cv::Point2d f0(timageS(wave.p, 1), timageS(wave.p, 0)), fn(timageS(neigh, 1), timageS(neigh, 0));
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
            
                cv::Point2d f1(timageS(p, 1), timageS(p, 0));
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

                            double fs = wave.fm0*t*t + wave.fm1*t + wave.fm2;

                            res1 = wave.m0*t*t + wave.m1*t + wave.m2 + cv::norm(dn - respos)*(fs+cv::norm(fn))/2.0;

                            if (res1 < y0) res1 = (1.0 - t)*y0 + t*y1 + cv::norm(dn - respos)*((1.0 - t)*cv::norm(f0) + t*cv::norm(f1)+cv::norm(fn))/2.0;
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
            
                cv::Point2d f1(timageS(p, 1), timageS(p, 0));
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

                            double fs = wave.fm3*t*t + wave.fm4*t + wave.fm5;

                            res2 = wave.m3*t*t + wave.m4*t + wave.m5 + cv::norm(dn - respos)*(fs+cv::norm(fn))/2.0;

                            if (res2 < y0) res2 = (1.0 - t)*y0 + t*y1 + cv::norm(dn - respos)*((1.0 - t)*cv::norm(f0) + t*cv::norm(f1)+cv::norm(fn))/2.0;
                        }
                    }
                }
            }
        }
        val = std::min(res1, res2);
        
    }
    
    return val;
    
}



double TurboWMMInterp::GoldenSearch(cv::Mat &image, cv::Mat &u_surface, TIsoWavefrontI &wave, cv::Point &neigh, cv::Point2d &h, bool forced) {
    
    cv::Point2d f0(timageS(wave.p, 1), timageS(wave.p, 0)), fn(timageS(neigh, 1), timageS(neigh, 0));
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
            
                cv::Point2d f1(timageS(p, 1), timageS(p, 0));
                if (std::isinf(cv::norm(f1)) || std::isnan(cv::norm(f1)))
                    f1 = fn;

                double a = 0.0, b = 1.0, x1 = a + (1-RESPHI)*(b - a), x2 = a + RESPHI*(b - a),
                f_x1 = MAX_VAL, f_x2 = MAX_VAL, i1, i2;

                cv::Point2d xtreme = dp + dd;

                double f_a = y0 + cv::norm(dn - dp)*(cv::norm(f0) + cv::norm(fn))/2.0;
                double f_b = y1 + cv::norm(dn - xtreme)*(cv::norm(f1) + cv::norm(fn))/2.0;

                res1 = (f_a < f_b) ? f_a : f_b;

                i1 = wave.fm0*x1*x1 + wave.fm1*x1 + wave.fm2;
                f_x1 = wave.m0*x1*x1 + wave.m1*x1 + wave.m2 + cv::norm(dn - (1.0 - x1)*dp - x1*xtreme)*(i1 + cv::norm(fn))/2.0;
                if (f_x1 < y0) f_x1 = (1.0 - x1)*y0 + x1*y1 + 
                        cv::norm(dn - (1.0 - x1)*dp - x1*xtreme)*((1.0 - x1)*cv::norm(f0) + x1*cv::norm(f1) + cv::norm(fn))/2.0;

                i2 = wave.fm0*x2*x2 + wave.fm1*x2 + wave.fm2;
                f_x2 = wave.m0*x2*x2 + wave.m1*x2 + wave.m2 + cv::norm(dn - (1.0 - x2)*dp - x2*xtreme)*(i2 + cv::norm(fn))/2.0;
                if (f_x2 < y0) f_x2 = (1.0 - x2)*y0 + x2*y1 + 
                        cv::norm(dn - (1.0 - x2)*dp - x2*xtreme)*((1.0 - x2)*cv::norm(f0) + x2*cv::norm(f1) + cv::norm(fn))/2.0;

                while (fabs(b - a) > TAU) {
                    if(f_x1 < f_x2) {
                        b = x2; x2 = x1; f_x2 = f_x1; x1 = a + (1 - RESPHI)*(b - a);
                        i1 = wave.fm0*x1*x1 + wave.fm1*x1 + wave.fm2;
                        f_x1 = wave.m0*x1*x1 + wave.m1*x1 + wave.m2 + cv::norm(dn - (1.0 - x1)*dp - x1*xtreme)*(i1 + cv::norm(fn))/2.0;
                        if (f_x1 < y0) f_x1 = (1.0 - x1)*y0 + x1*y1 + 
                                cv::norm(dn - (1.0 - x1)*dp - x1*xtreme)*((1.0 - x1)*cv::norm(f0) + x1*cv::norm(f1) + cv::norm(fn))/2.0;
                    }
                    else {
                        a = x1; x1 = x2; f_x1 = f_x2; x2 = a + RESPHI*(b - a);
                        i2 = wave.fm0*x2*x2 + wave.fm1*x2 + wave.fm2;
                        f_x2 = wave.m0*x2*x2 + wave.m1*x2 + wave.m2 + cv::norm(dn - (1.0 - x2)*dp - x2*xtreme)*(i2 + cv::norm(fn))/2.0;
                        if (f_x2 < y0) f_x2 = (1.0 - x2)*y0 + x2*y1 + 
                                cv::norm(dn - (1.0 - x2)*dp - x2*xtreme)*((1.0 - x2)*cv::norm(f0) + x2*cv::norm(f1) + cv::norm(fn))/2.0;
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
            
                cv::Point2d f1(timageS(p, 1), timageS(p, 0));
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


                i1 = wave.fm3*x1*x1 + wave.fm4*x1 + wave.fm5;
                f_x1 = wave.m3*x1*x1 + wave.m4*x1 + wave.m5 + cv::norm(dn - (1.0 - x1)*dp - x1*xtreme)*(i1 + cv::norm(fn))/2.0;
                if (f_x1 < y0) f_x1 = (1.0 - x1)*y0 + x1*y1 + 
                        cv::norm(dn - (1.0 - x1)*dp - x1*xtreme)*((1.0 - x1)*cv::norm(f0) + x1*cv::norm(f1) + cv::norm(fn))/2.0;

                i2 = wave.fm3*x2*x2 + wave.fm4*x2 + wave.fm5;
                f_x2 = wave.m3*x2*x2 + wave.m4*x2 + wave.m5 + cv::norm(dn - (1.0 - x2)*dp - x2*xtreme)*(i2 + cv::norm(fn))/2.0;
                if (f_x2 < y0) f_x2 = (1.0 - x2)*y0 + x2*y1 + 
                        cv::norm(dn - (1.0 - x2)*dp - x2*xtreme)*((1.0 - x2)*cv::norm(f0) + x2*cv::norm(f1) + cv::norm(fn))/2.0;

                while (fabs(b - a) > TAU) {
                    if(f_x1 < f_x2) {
                        b = x2; x2 = x1; f_x2 = f_x1; x1 = a + (1 - RESPHI)*(b - a);
                        i1 = wave.fm3*x1*x1 + wave.fm4*x1 + wave.fm5;
                        f_x1 = wave.m3*x1*x1 + wave.m4*x1 + wave.m5 + cv::norm(dn - (1.0 - x1)*dp - x1*xtreme)*(i1 + cv::norm(fn))/2.0;
                        if (f_x1 < y0) f_x1 = (1.0 - x1)*y0 + x1*y1 + 
                                cv::norm(dn - (1.0 - x1)*dp - x1*xtreme)*((1.0 - x1)*cv::norm(f0) + x1*cv::norm(f1) + cv::norm(fn))/2.0;
                    }
                    else {
                        a = x1; x1 = x2; f_x1 = f_x2; x2 = a + RESPHI*(b - a);
                        i2 = wave.fm3*x2*x2 + wave.fm4*x2 + wave.fm5;
                        f_x2 = wave.m3*x2*x2 + wave.m4*x2 + wave.m5 + cv::norm(dn - (1.0 - x2)*dp - x2*xtreme)*(i2 + cv::norm(fn))/2.0;
                        if (f_x2 < y0) f_x2 = (1.0 - x2)*y0 + x2*y1 + 
                                cv::norm(dn - (1.0 - x2)*dp - x2*xtreme)*((1.0 - x2)*cv::norm(f0) + x2*cv::norm(f1) + cv::norm(fn))/2.0;
                    }
                }
                res2 = std::min(res2, std::min(f_x1, f_x2));
            }
        }
        val = std::min(res1, res2);
        
    }
    
    return val;
    
}



void TurboWMMInterp::getInterp(double *y, TripleI *m, int pos) {
    
    if (pos%2 == 0) {
        m[pos].p2 = y[pos];
        m[pos].p1 = (4.0*y[(pos+1)%8] - y[(pos+2)%8] - 3.0*m[pos].p2)/2.0;
        m[pos].p0 = y[(pos+1)%8] - m[pos].p1 - m[pos].p2;
    }
    else {
        m[pos].p2 = y[pos];
        m[pos].p1 = (y[(pos+1)%8] - y[(pos+7)%8])/2.0;
        m[pos].p0 = y[(pos+7)%8] + m[pos].p1 - m[pos].p2;
    }

}


void TurboWMMInterp::getInterpInverse(double *y, TripleI *m, int pos) {
    
    if (pos%2 == 0) {
        m[pos].p2 = y[pos];
        m[pos].p1 = (4.0*y[(pos+7)%8] - y[(pos+6)%8] - 3.0*m[pos].p2)/2.0;
        m[pos].p0 = y[(pos+7)%8] - m[pos].p1 - m[pos].p2;
    }
    else {
        m[pos].p2 = y[pos];
        m[pos].p1 = (y[(pos+7)%8] - y[(pos+1)%8])/2.0;
        m[pos].p0 = y[(pos+1)%8] + m[pos].p1 - m[pos].p2;
    }

}


TurboWMMInterp::~TurboWMMInterp() {
    
}


int TurboWMMInterp::solveLinear(double *C, double *S) {
    S[0] = -C[0]/C[1];
    return true;
}

int TurboWMMInterp::solveQuadric(double *C, double *S) {
    
    double p, q, D;

    if (fabs(C[2]) < TAU)
        return solveLinear(C, S);

    p = C[1] / (2.0 * C[2]);
    q = C[0] / C[2];
    D = p * p - q;

    if (fabs(D) < TAU) {
        S[0] = -p;
        return 1;
    }

    if (D < 0.0)
        return 0;
    else {
        double sqrt_D = sqrt(D);
        S[0] = sqrt_D - p;
        S[1] = -sqrt_D - p;
        return 2;
    }
    
}

int TurboWMMInterp::solveCubic(double *C, double *S) {
    
    int i, num;
    double sub, A, B, C1, sq_A, p, q, cb_p, D;

    // normalize the equation:x ^ 3 + Ax ^ 2 + Bx  + C = 0
    A = C[2] / C[3];
    B = C[1] / C[3];
    C1 = C[0] / C[3];

    // substitute x = y - A / 3 to eliminate the quadric term: x^3 + px + q = 0
    sq_A = A * A;
    p = 1.0/3.0 * (-1.0/3.0 * sq_A + B);
    q = 1.0/2.0 * (2.0/27.0 * A *sq_A - 1.0/3.0 * A * B + C1);

    // use Cardano's formula
    cb_p = p * p * p;
    D = q * q + cb_p;

    if (fabs(D) < TAU) {
        if (fabs(q) < TAU) {
            S[0] = 0.0;
            num = 1;
        }
        else {
            double u = cbrt(-q);
            S[0] = 2.0 * u;
            S[1] = - u;
            num = 2;
        }
    }
    else {
        if (D < 0.0) {
            double phi = 1.0/3.0 * acos(-q / sqrt(-cb_p));
            double t = 2.0 * sqrt(-p);
            S[0] = t * cos(phi);
            S[1] = -t * cos(phi + M_PI / 3.0);
            S[2] = -t * cos(phi - M_PI / 3.0);
            num = 3;
        }
        else {
            double sqrt_D = sqrt(D);
            double u = cbrt(sqrt_D + fabs(q));
            if (q > 0.0)
                S[0] = - u + p / u ;
            else
                S[0] = u - p / u;
            num = 1;
        }
    }

    // resubstitute
    sub = 1.0 / 3.0 * A;
    for (i = 0; i < num; i++)
        S[i] -= sub;
    return num;
}

int TurboWMMInterp::solveQuartic(double *C, double *S) {
    
    double coeffs[4], z, u, v, sub, A, B, C1, D, sq_A, p, q, r;
    int i, num;

    // normalize the equation:x ^ 4 + Ax ^ 3 + Bx ^ 2 + Cx + D = 0
    A = C[3] / C[4];
    B = C[2] / C[4];
    C1 = C[1] / C[4];
    D = C[0] / C[4];

    // subsitute x = y - A / 4 to eliminate the cubic term: x^4 + px^2 + qx + r = 0
    sq_A = A * A;
    p = -3.0 / 8.0 * sq_A + B;
    q = 1.0 / 8.0 * sq_A * A - 1.0 / 2.0 * A * B + C1;
    r = -3.0 / 256.0 * sq_A * sq_A + 1.0 / 16.0 * sq_A * B - 1.0 / 4.0 * A * C1 + D;

    if (fabs(r) < TAU) {
        // no absolute term:y(y ^ 3 + py + q) = 0
        coeffs[0] = q;
        coeffs[1] = p;
        coeffs[2] = 0.0;
        coeffs[3] = 1.0;
        num = solveCubic(coeffs, S);
        S[num++] = 0;
    }
    else {
        // solve the resolvent cubic...
        coeffs[0] = 1.0 / 2.0 * r * p - 1.0 / 8.0 * q * q;
        coeffs[1] = -r;
        coeffs[2] = -1.0 / 2.0 * p;
        coeffs[3] = 1.0;
        (void) solveCubic(coeffs, S);

        // ...and take the one real solution...
        z = S[0];

        // ...to build two quadratic equations
        u = z * z - r;
        v = 2.0 * z - p;

        if (fabs(u) < TAU)
            u = 0.0;
        else if (u > 0.0)
            u = sqrt(u);
        else
            return 0;

        if (fabs(v) < TAU)
            v = 0;
        else if (v > 0.0)
            v = sqrt(v);
        else
            return 0;

        coeffs[0] = z - u;
        coeffs[1] = q < 0 ? -v : v;
        coeffs[2] = 1.0;

        num = solveQuadric(coeffs, S);

        coeffs[0] = z + u;
        coeffs[1] = q < 0 ? v : -v;
        coeffs[2] = 1.0;

        num += solveQuadric(coeffs, S + num);
    }

    // resubstitute
    sub = 1.0 / 4 * A;
    for (i = 0; i < num; i++)
        S[i] -= sub;

    return num;

}

