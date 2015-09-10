clc
clear all
close all

addpath(genpath('..'));

h = [.1, .1];
tam = 500;

initials = [251, 251];

ispeed2 = zeros(tam, tam, 2);
ispeed = zeros(tam);
gt = zeros(tam);

for i=1:tam
    for j=1:tam
        diff = h.*([i, j] - initials);
        gt(i,j) = 1 - cos(diff(2)/20)*cos(diff(1)/20);
        dx = 1/20*sin(diff(2)/20)*cos(diff(1)/20);
        dy = 1/20*cos(diff(2)/20)*sin(diff(1)/20);
        ispeed(i,j) = 1/sqrt(dx*dx + dy*dy);
        ispeed2(i,j,1) = dy;
        ispeed2(i,j,2) = dx;
    end
end

it = 10;
DispResults;