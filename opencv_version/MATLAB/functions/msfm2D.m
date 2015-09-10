function [ surface ] = msfm2D(potential, initials, h, order)
% 
% Usage:
%
% [SURFACE] = msfm2D(POTENTIAL, INITIALS, H, ORDER)
%
% Parameters:
%       POTENTIAL:: A MxN matrix which specifies the potential of each node
%
%       INITIALS:: A Lx2 matrix specifying the initial points.
%
%       H:: 1x2 vector. Distance between nodes (y_distance, x_distance).
%
%       ORDER:: approximation order (1 or 2)
%
% Outputs:
%       SURFACE:: A MxN matrix with the solution.
%
surface = msfm2Dc(potential, initials, h, order);
