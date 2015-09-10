#include "../header/MSFM.h"
#include <opencv/cv.h>
#include <map>

#define MAX_VAL 100000
#define P_FAR   0
#define P_ALIVE 1
#define P_TRIAL 2

#define image2M(p) reinterpret_cast<double *>(image.data)[(int) (p.y + image.rows*p.x)]
#define state2M(p) state.data[(int) (p.y + state.rows*p.x)]
#define distance2M(p) reinterpret_cast<double *>(u_surface.data)[(int) (p.y + u_surface.rows*p.x)]
#define contains2M(p) ((p.x < image.cols) && (p.y < image.rows) && (p.x >= 0) && (p.y >= 0))

cv::Mat MSFM::MSFMSurfaceO1(cv::Mat& image, cv::vector<cv::Point>& initials, cv::Point2d &h) {
    
    cv::Mat u_surface = MAX_VAL*cv::Mat::ones(image.rows, image.cols, CV_64FC1);
    cv::Mat state = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
    
    std::multimap<double, cv::Point> trial_set;
    std::map<int, std::multimap<double, cv::Point>::iterator> mapa_trial;
    
    std::multimap<double, cv::Point>::iterator trial_set_it;
    std::map<int, std::multimap<double, cv::Point>::iterator>::iterator mapa_trial_it;
    std::pair<double, cv::Point> pr_trial;
    std::pair<int, std::multimap<double, cv::Point>::iterator> pr_mapa;
    int key, i;
    cv::Point winner, neigh;
    
    // Initialization
    for (i = 0; i < (int) initials.size(); i++) {
        key = initials[i].y + image.rows*initials[i].x;
        if (mapa_trial.find(key) == mapa_trial.end()) {
            distance2M(initials[i]) = 0.0;
            state2M(initials[i]) = P_TRIAL;
            pr_trial = std::pair<double, cv::Point>(0.0, initials[i]);
            trial_set_it = trial_set.insert(pr_trial);
            pr_mapa = std::pair<int, std::multimap<double, cv::Point>::iterator>(key, trial_set_it);
            mapa_trial.insert(pr_mapa);
        }
    }
        
    // LOOP
    while (!trial_set.empty()) {
        
        trial_set_it = trial_set.begin();
        
        key = trial_set_it->second.y + image.rows*trial_set_it->second.x;
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
                
        state2M(winner) = P_ALIVE;
        
        // UPWIND PROCEDURE
        for (int i=-1; i<2; i+=2) {
            neigh = cv::Point(winner.x + i, winner.y);
            if (contains2M(neigh))
                this->StencilS1O1(image, u_surface, state, trial_set, mapa_trial, neigh, h);
            neigh = cv::Point(winner.x, winner.y + i);
            if (contains2M(neigh))
                this->StencilS1O1(image, u_surface, state, trial_set, mapa_trial, neigh, h);
            neigh = cv::Point(winner.x + i, winner.y + i);
            if (contains2M(neigh))
                this->StencilS2O1(image, u_surface, state, trial_set, mapa_trial, neigh, h);
            neigh = cv::Point(winner.x - i, winner.y + i);
            if (contains2M(neigh))
                this->StencilS2O1(image, u_surface, state, trial_set, mapa_trial, neigh, h);
        }
        
    }
    
    return u_surface;
}


cv::Mat MSFM::MSFMSurfaceO2(cv::Mat& image, cv::vector<cv::Point>& initials, cv::Point2d &h) {
    
    cv::Mat u_surface = MAX_VAL*cv::Mat::ones(image.rows, image.cols, CV_64FC1);
    cv::Mat state = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
    
    std::multimap<double, cv::Point> trial_set;
    std::map<int, std::multimap<double, cv::Point>::iterator> mapa_trial;
    
    std::multimap<double, cv::Point>::iterator trial_set_it;
    std::map<int, std::multimap<double, cv::Point>::iterator>::iterator mapa_trial_it;
    std::pair<double, cv::Point> pr_trial;
    std::pair<int, std::multimap<double, cv::Point>::iterator> pr_mapa;
    int key, i;
    cv::Point winner, neigh;
    
    // Initialization
    for (i = 0; i < (int) initials.size(); i++) {
        key = initials[i].y + image.rows*initials[i].x;
        if (mapa_trial.find(key) == mapa_trial.end()) {
            distance2M(initials[i]) = 0.0;
            state2M(initials[i]) = P_TRIAL;
            pr_trial = std::pair<double, cv::Point>(0.0, initials[i]);
            trial_set_it = trial_set.insert(pr_trial);
            pr_mapa = std::pair<int, std::multimap<double, cv::Point>::iterator>(key, trial_set_it);
            mapa_trial.insert(pr_mapa);
        }
    }
    
    // LOOP
    while (!trial_set.empty()) {
        
        trial_set_it = trial_set.begin();
        
        key = trial_set_it->second.y + image.rows*trial_set_it->second.x;
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
                
        state2M(winner) = P_ALIVE;
        
        // UPWIND PROCEDURE
        for (int i=-1; i<2; i+=2) {
            neigh = cv::Point(winner.x + i, winner.y);
            if (contains2M(neigh))
                this->StencilS1O2(image, u_surface, state, trial_set, mapa_trial, neigh, h);
            neigh = cv::Point(winner.x, winner.y + i);
            if (contains2M(neigh))
                this->StencilS1O2(image, u_surface, state, trial_set, mapa_trial, neigh, h);
            neigh = cv::Point(winner.x + i, winner.y + i);
            if (contains2M(neigh))
                this->StencilS2O2(image, u_surface, state, trial_set, mapa_trial, neigh, h);
            neigh = cv::Point(winner.x - i, winner.y + i);
            if (contains2M(neigh))
                this->StencilS2O2(image, u_surface, state, trial_set, mapa_trial, neigh, h);
        }
    }
    
    return u_surface;
}


MSFM::~MSFM() {
}

void MSFM::StencilS1O1(cv::Mat& image, cv::Mat& u_surface, cv::Mat& state, std::multimap<double,cv::Point>& trial_set, 
        std::map<int,std::multimap<double,cv::Point>::iterator>& mapa_trial, cv::Point& neigh, cv::Point2d &h) {
    
    unsigned char neigh_state = state2M(neigh);
    
    if (neigh_state == P_ALIVE)
        return;
    
    int key = neigh.y + image.rows*neigh.x;
    
    double F = image2M(neigh);
    
    double T_2a = MAX_VAL, T_2b = MAX_VAL, T_1a = MAX_VAL, T_1b = MAX_VAL, gh1, gh2;
    cv::Point p;
    
    gh1 = 1.0/(h.x*h.x);
    p = cv::Point(neigh.x - 1, neigh.y);
    if (contains2M(p)) {
        if (state2M(p) == P_ALIVE)
            T_1a = distance2M(p);
    }
    p = cv::Point(neigh.x + 1, neigh.y);
    if (contains2M(p)) {
        if (state2M(p) == P_ALIVE)
            T_1b = distance2M(p);
    }
    
    gh2 = 1.0/(h.y*h.y);
    p = cv::Point(neigh.x, neigh.y - 1);
    if (contains2M(p)) {
        if (state2M(p) == P_ALIVE)
            T_2a = distance2M(p);
    }
    p = cv::Point(neigh.x, neigh.y + 1);
    if (contains2M(p)) {
        if (state2M(p) == P_ALIVE)
            T_2b = distance2M(p);
    }
    
    double T_1 = std::min(T_1a, T_1b);
    double T_2 = std::min(T_2a, T_2b);
    
    double new_val;
    
    double a_1 = gh1 + gh2;
    double b_1 = -2.0*(T_1*gh1 + T_2*gh2);
    double c_1 = T_1*T_1*gh1 + T_2*T_2*gh2 - 1.0/(F*F);
    double disc = b_1*b_1 - 4.0*a_1*c_1;
    new_val = -1.0;
    if (disc >= 0.0) {
        new_val = (-b_1 + sqrt(disc))/(2.0*a_1);
        if (new_val < std::min(T_1, T_2))
            new_val = MAX_VAL;
    } 
    if ((new_val <= T_1 && T_1 < MAX_VAL) || (new_val <= T_2 && T_2 < MAX_VAL) || new_val >= MAX_VAL) {
        double s1 = T_1 + 1.0/(F*sqrt(gh1)), s2 = T_2 + 1.0/(F*sqrt(gh2));
        new_val = std::min(s1, s2);
    }
        
    if (neigh_state == P_TRIAL) {
        if (new_val < distance2M(neigh)) {
            std::map<int, std::multimap<double, cv::Point>::iterator>::iterator mapa_trial_it;
            mapa_trial_it = mapa_trial.find(key);
            trial_set.erase(mapa_trial_it->second);
            mapa_trial.erase(mapa_trial_it);
        }
        else 
            return;
    }
    else
        state2M(neigh) = P_TRIAL;
    
    std::multimap<double, cv::Point>::iterator trial_set_it;
    std::pair<double, cv::Point> pr_trial(new_val, neigh);
    trial_set_it = trial_set.insert(pr_trial);
    std::pair<int, std::multimap<double, cv::Point>::iterator> pr_mapa(key, trial_set_it);
    mapa_trial.insert(pr_mapa);
    distance2M(neigh) = new_val;
        
}



void MSFM::StencilS1O2(cv::Mat& image, cv::Mat& u_surface, cv::Mat& state, std::multimap<double,cv::Point>& trial_set, 
        std::map<int,std::multimap<double,cv::Point>::iterator>& mapa_trial, cv::Point& neigh, cv::Point2d &h) {
    
    unsigned char neigh_state = state2M(neigh);
    
    if (neigh_state == P_ALIVE)
        return;
    
    int key = neigh.y + image.rows*neigh.x;
    
    double F = image2M(neigh);
    
    double T_2a = MAX_VAL, T_2b = MAX_VAL, T_1a = MAX_VAL, T_1b = MAX_VAL, gh1, gh2;
    double T_21a = MAX_VAL, T_21b = MAX_VAL, T_11a = MAX_VAL, T_11b = MAX_VAL;
    bool T_1n = false, T_2n = false;
    cv::Point p;
    
    gh1 = 1.0/(h.x*h.x);
    p = cv::Point(neigh.x - 1, neigh.y);
    if (contains2M(p)) {
        if (state2M(p) == P_ALIVE) {
            T_1a = distance2M(p);
            p = cv::Point(neigh.x - 2, neigh.y);
            if (contains2M(p)) {
                T_11a = distance2M(p);
                if (state2M(p) == P_ALIVE)
                    T_1n = true;
            }
        }
    }
    p = cv::Point(neigh.x + 1, neigh.y);
    if (contains2M(p)) {
        if (state2M(p) == P_ALIVE) {
            T_1b = distance2M(p);
            p = cv::Point(neigh.x + 2, neigh.y);
            if (contains2M(p)) {
                T_11b = distance2M(p);
                if (state2M(p) == P_ALIVE)
                    T_1n = true;
            }
        }
    }
    
    gh2 = 1.0/(h.y*h.y);
    p = cv::Point(neigh.x, neigh.y + 1);
    if (contains2M(p)) {
        if (state2M(p) == P_ALIVE) {
            T_2a = distance2M(p);
            p = cv::Point(neigh.x, neigh.y + 2);
            if (contains2M(p)) {
                T_21a = distance2M(p);
                if (state2M(p) == P_ALIVE)
                    T_2n = true;
            }
        }
    }
    p = cv::Point(neigh.x, neigh.y - 1);
    if (contains2M(p)) {
        if (state2M(p) == P_ALIVE) {
            T_2b = distance2M(p);
            p = cv::Point(neigh.x, neigh.y - 2);
            if (contains2M(p)) {
                T_21b = distance2M(p);
                if (state2M(p) == P_ALIVE)
                    T_2n = true;
            }
        }
    }
    
    int order = (T_1n && T_2n && ((T_1a >= T_11a && T_1a <= T_1b) || (T_1b >= T_11b && T_1b <= T_1a)) && 
        ((T_2a >= T_21a && T_2a <= T_2b) || (T_2b >= T_21b && T_2b <= T_2a))) ? 2 : 1;
    double T_1, T_2;
    switch (order) {
        case 1:
            T_1 = std::min(T_1a, T_1b);
            T_2 = std::min(T_2a, T_2b);
            break;
        case 2:
            if (T_1a >= T_11a && T_1b >= T_11b)
                T_1 = std::min((4.0*T_1a - T_11a)/3.0, (4.0*T_1b - T_11b)/3.0);
            else if (T_1a >= T_11a)
                T_1 = (4.0*T_1a - T_11a)/3.0;
            else
                T_1 = (4.0*T_1b - T_11b)/3.0;
            if (T_2a >= T_21a && T_2b >= T_21b)
                T_2 = std::min((4.0*T_2a - T_21a)/3.0, (4.0*T_2b - T_21b)/3.0);
            else if (T_2a >= T_21a)
                T_2 = (4.0*T_2a - T_21a)/3.0;
            else
                T_2 = (4.0*T_2b - T_21b)/3.0;
            gh1 *= 9.0/4.0;
            gh2 *= 9.0/4.0;
            break;
    }
    
    if (T_1 == MAX_VAL && T_2 == MAX_VAL)
        return;
    
    
    double new_val;
    
    double a_1 = gh1 + gh2;
    double b_1 = -2.0*(T_1*gh1 + T_2*gh2);
    double c_1 = T_1*T_1*gh1 + T_2*T_2*gh2 - 1.0/(F*F);
    double disc = b_1*b_1 - 4.0*a_1*c_1;
    new_val = -1.0;
    if (disc >= 0.0) {
        new_val = (-b_1 + sqrt(disc))/(2.0*a_1);
        if (new_val < std::min(T_1, T_2))
            new_val = MAX_VAL;
    } 
    if ((new_val <= T_1 && T_1 < MAX_VAL) || (new_val <= T_2 && T_2 < MAX_VAL) || new_val >= MAX_VAL) {
        double s1 = T_1 + 1.0/(F*sqrt(gh1)), s2 = T_2 + 1.0/(F*sqrt(gh2));
        new_val = std::min(s1, s2);
    }
    
    if (neigh_state == P_TRIAL) {
        if (new_val < distance2M(neigh)) {
            std::map<int, std::multimap<double, cv::Point>::iterator>::iterator mapa_trial_it;
            mapa_trial_it = mapa_trial.find(key);
            trial_set.erase(mapa_trial_it->second);
            mapa_trial.erase(mapa_trial_it);
        }
        else 
            return;
    }
    else
        state2M(neigh) = P_TRIAL;
    
    std::multimap<double, cv::Point>::iterator trial_set_it;
    std::pair<double, cv::Point> pr_trial(new_val, neigh);
    trial_set_it = trial_set.insert(pr_trial);
    std::pair<int, std::multimap<double, cv::Point>::iterator> pr_mapa(key, trial_set_it);
    mapa_trial.insert(pr_mapa);
    distance2M(neigh) = new_val;
        
}



void MSFM::StencilS2O1(cv::Mat& image, cv::Mat& u_surface, cv::Mat& state, std::multimap<double,cv::Point>& trial_set, 
        std::map<int,std::multimap<double,cv::Point>::iterator>& mapa_trial, cv::Point& neigh, cv::Point2d &h) {
    
    unsigned char neigh_state = state2M(neigh);
    
    if (neigh_state == P_ALIVE)
        return;
    
    int key = neigh.y + image.rows*neigh.x;
    
    double F = image2M(neigh);
    
    double T_2a = MAX_VAL, T_2b = MAX_VAL, T_1a = MAX_VAL, T_1b = MAX_VAL, gh1, gh2;
    cv::Point p;
    
    gh1 = 1.0/(h.x*h.x + h.y*h.y);
    p = cv::Point(neigh.x - 1, neigh.y - 1);
    if (contains2M(p)) {
        if (state2M(p) == P_ALIVE)
            T_1a = distance2M(p);
    }
    p = cv::Point(neigh.x + 1, neigh.y + 1);
    if (contains2M(p)) {
        if (state2M(p) == P_ALIVE)
            T_1b = distance2M(p);
    }
    
    gh2 = 1.0/(h.x*h.x + h.y*h.y);
    p = cv::Point(neigh.x - 1, neigh.y + 1);
    if (contains2M(p)) {
        if (state2M(p) == P_ALIVE)
            T_2a = distance2M(p);
    }
    p = cv::Point(neigh.x + 1, neigh.y - 1);
    if (contains2M(p)) {
        if (state2M(p) == P_ALIVE)
            T_2b = distance2M(p);
    }
    
    double T_1 = std::min(T_1a, T_1b);
    double T_2 = std::min(T_2a, T_2b);
    
    double cos_phi = fabs(h.x*h.x - h.y*h.y)/(h.x*h.x + h.y*h.y);
    double sin_phi = sqrt(1.0 - cos_phi*cos_phi);
    
    double new_val;
    
    double a_1 = gh1 + gh2 - 2.0*cos_phi/(h.x*h.y);
    double b_1 = -2.0*(T_1*gh1 + T_2*gh2) + 2.0*cos_phi*(T_1 + T_2)/(h.x*h.y);
    double c_1 = T_1*T_1*gh1 + T_2*T_2*gh2 - sin_phi*sin_phi/(F*F) - 2.0*cos_phi*T_1*T_2/(h.x*h.y);
    double disc = b_1*b_1 - 4.0*a_1*c_1;
    new_val = -1.0;
    if (disc >= 0.0) {
        new_val = (-b_1 + sqrt(disc))/(2.0*a_1);
        if (new_val < std::min(T_1, T_2))
            new_val = MAX_VAL;
        
    } 
    if ((new_val <= T_1 && T_1 < MAX_VAL) || (new_val <= T_2 && T_2 < MAX_VAL) || new_val >= MAX_VAL) {
        double s1 = T_1 + 1.0/(F*sqrt(gh1)), s2 = T_2 + 1.0/(F*sqrt(gh2));
        new_val = std::min(s1, s2);
    }
    
    if (neigh_state == P_TRIAL) {
        if (new_val < distance2M(neigh)) {
            std::map<int, std::multimap<double, cv::Point>::iterator>::iterator mapa_trial_it;
            mapa_trial_it = mapa_trial.find(key);
            trial_set.erase(mapa_trial_it->second);
            mapa_trial.erase(mapa_trial_it);
        }
        else 
            return;
    }
    else
        state2M(neigh) = P_TRIAL;
    
    std::multimap<double, cv::Point>::iterator trial_set_it;
    std::pair<double, cv::Point> pr_trial(new_val, neigh);
    trial_set_it = trial_set.insert(pr_trial);
    std::pair<int, std::multimap<double, cv::Point>::iterator> pr_mapa(key, trial_set_it);
    mapa_trial.insert(pr_mapa);
    distance2M(neigh) = new_val;
}


void MSFM::StencilS2O2(cv::Mat& image, cv::Mat& u_surface, cv::Mat& state, std::multimap<double,cv::Point>& trial_set, 
        std::map<int,std::multimap<double,cv::Point>::iterator>& mapa_trial, cv::Point& neigh, cv::Point2d &h) {
    
    unsigned char neigh_state = state2M(neigh);
    
    if (neigh_state == P_ALIVE)
        return;
    
    int key = neigh.y + image.rows*neigh.x;
    
    double F = image2M(neigh);
    
    double T_2a = MAX_VAL, T_2b = MAX_VAL, T_1a = MAX_VAL, T_1b = MAX_VAL, gh1, gh2;
    double T_21a = MAX_VAL, T_21b = MAX_VAL, T_11a = MAX_VAL, T_11b = MAX_VAL;
    bool T_1n = false, T_2n = false;
    cv::Point p;
    
    
    gh1 = 1.0/(h.x*h.x + h.y*h.y);
    p = cv::Point(neigh.x - 1, neigh.y - 1);
    if (contains2M(p)) {
        if (state2M(p) == P_ALIVE) {
            T_1a = distance2M(p);
            p = cv::Point(neigh.x - 2, neigh.y - 2);
            if (contains2M(p)) {
                T_11a = distance2M(p);
                if (state2M(p) == P_ALIVE)
                    T_1n = true;
            }
        }
    }
    p = cv::Point(neigh.x + 1, neigh.y + 1);
    if (contains2M(p)) {
        if (state2M(p) == P_ALIVE) {
            T_1b = distance2M(p);
            p = cv::Point(neigh.x + 2, neigh.y + 2);
            if (contains2M(p)) {
                T_11b = distance2M(p);
                if (state2M(p) == P_ALIVE)
                    T_1n = true;
            }
        }
    }
    
    gh2 = 1.0/(h.x*h.x + h.y*h.y);
    p = cv::Point(neigh.x - 1, neigh.y + 1);
    if (contains2M(p)) {
        if (state2M(p) == P_ALIVE) {
            T_2a = distance2M(p);
            p = cv::Point(neigh.x - 2, neigh.y + 2);
            if (contains2M(p)) {
                T_21a = distance2M(p);
                if (state2M(p) == P_ALIVE)
                    T_2n = true;
            }
        }
    }
    p = cv::Point(neigh.x + 1, neigh.y - 1);
    if (contains2M(p)) {
        if (state2M(p) == P_ALIVE) {
            T_2b = distance2M(p);
            p = cv::Point(neigh.x + 2, neigh.y - 2);
            if (contains2M(p)) {
                T_21b = distance2M(p);
                if (state2M(p) == P_ALIVE)
                    T_2n = true;
            }
        }
    }
    
    
    int order = (T_1n && T_2n && ((T_1a >= T_11a && T_1a <= T_1b) || (T_1b >= T_11b && T_1b <= T_1a)) && 
        ((T_2a >= T_21a && T_2a <= T_2b) || (T_2b >= T_21b && T_2b <= T_2a))) ? 2 : 1;
    
    double T_1, T_2;
    switch (order) {
        case 1:
            T_1 = std::min(T_1a, T_1b);
            T_2 = std::min(T_2a, T_2b);
            break;
        case 2:
            if (T_1a >= T_11a && T_1b >= T_11b)
                T_1 = std::min((4.0*T_1a - T_11a)/3.0, (4.0*T_1b - T_11b)/3.0);
            else if (T_1a >= T_11a)
                T_1 = (4.0*T_1a - T_11a)/3.0;
            else
                T_1 = (4.0*T_1b - T_11b)/3.0;
            if (T_2a >= T_21a && T_2b >= T_21b)
                T_2 = std::min((4.0*T_2a - T_21a)/3.0, (4.0*T_2b - T_21b)/3.0);
            else if (T_2a >= T_21a)
                T_2 = (4.0*T_2a - T_21a)/3.0;
            else
                T_2 = (4.0*T_2b - T_21b)/3.0;
            gh1 *= 9.0/4.0;
            gh2 *= 9.0/4.0;
            break;
    }
    
    double cos_phi = fabs(h.x*h.x - h.y*h.y)/(h.x*h.x + h.y*h.y);
    double sin_phi = sqrt(1.0 - cos_phi*cos_phi);
    
    double new_val;
    
    double a_1 = gh1 + gh2 - 2.0*cos_phi/(h.x*h.y);
    double b_1 = -2.0*(T_1*gh1 + T_2*gh2) + 2.0*cos_phi*(T_1 + T_2)/(h.x*h.y);
    double c_1 = T_1*T_1*gh1 + T_2*T_2*gh2 - sin_phi*sin_phi/(F*F) - 2.0*cos_phi*T_1*T_2/(h.x*h.y);
    double disc = b_1*b_1 - 4.0*a_1*c_1;
    new_val = -1.0;
    if (disc >= 0.0) {
        new_val = (-b_1 + sqrt(disc))/(2.0*a_1);
        if (new_val < std::min(T_1, T_2))
            new_val = MAX_VAL;
    } 
    if ((new_val <= T_1 && T_1 < MAX_VAL) || (new_val <= T_2 && T_2 < MAX_VAL) || new_val >= MAX_VAL) {
        double s1 = T_1 + 1.0/(F*sqrt(gh1)), s2 = T_2 + 1.0/(F*sqrt(gh2));
        new_val = std::min(s1, s2);
    }
    
    if (neigh_state == P_TRIAL) {
        if (new_val < distance2M(neigh)) {
            std::map<int, std::multimap<double, cv::Point>::iterator>::iterator mapa_trial_it;
            mapa_trial_it = mapa_trial.find(key);
            trial_set.erase(mapa_trial_it->second);
            mapa_trial.erase(mapa_trial_it);
        }
        else 
            return;
    }
    else
        state2M(neigh) = P_TRIAL;
    
    std::multimap<double, cv::Point>::iterator trial_set_it;
    std::pair<double, cv::Point> pr_trial(new_val, neigh);
    trial_set_it = trial_set.insert(pr_trial);
    std::pair<int, std::multimap<double, cv::Point>::iterator> pr_mapa(key, trial_set_it);
    mapa_trial.insert(pr_mapa);
    distance2M(neigh) = new_val;
        
}
