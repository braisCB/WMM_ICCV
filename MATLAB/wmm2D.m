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
%       IFUNC::     Interpolation function. Options:
%                           'linear'
%                           'spline'
%                           'hermite'
%                           'pchip'
%                           'quad'
%
%       SFUNC::     Search function. Options:
%                           'gradient'      -> Gradient function
%                           'hopf_lax'      -> Modified Hopf-Lax function
%                           'golden_search' -> Golden Search function  
%
% Outputs:
%       SURFACE::   A MxN matrix with the solution.
%


addpath('./mex');

[ surface ] = wmm2D_c(potential, initials, h, sfunc, ifunc);