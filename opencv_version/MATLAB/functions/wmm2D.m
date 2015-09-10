function [ surface ] = wmm2D(potential, initials, h, sfunc, ifunc)
% 
% Usage:
%
% [SURFACE] = wmm2D(POTENTIAL, INITIALS, H, SFUNC, IFUNC)
%
% Parameters:
%       POTENTIAL:: A MxNx2 matrix which specifies the potential of each node
%
%       INITIALS::  A Lx2 matrix specifying the initial points.
%
%       H::         1x2 vector. Distance between nodes (y_distance, x_distance).
%
%       SFUNC::     Search function. Options:
%                           'gradient'      -> Gradient function
%                           'hopf_lax'      -> Modified Hopf-Lax function
%                           'golden_search' -> Golden Search function  
%
%       IFUNC::     Interpolation function. Options:
%                           'linear'
%                           'spline'
%                           'hermite'
%                           'pchip'
%                           'quadric'
%
% Outputs:
%       SURFACE::   A MxN matrix with the solution.
%

switch ifunc
    case 'linear'
        surface = wmmTurboLinear2D(potential, initials, h, sfunc);
    case 'spline'
        surface = wmmTurboSpline2D(potential, initials, h, sfunc);
    case 'hermite'
        surface = wmmTurboHermite2D(potential, initials, h, sfunc);
    case 'pchip'
        surface = wmmTurboPchip2D(potential, initials, h, sfunc);
    case 'quadric'
        surface = wmmTurboInterp2D(potential, initials, h, sfunc);
    otherwise
        error('Incorrect Search function. See "help wmm2D"');
end
