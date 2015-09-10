clc
clear all
close all

addpath(genpath('..'));

h = [1, 1];
tam = 101;

initials = (tam+1)/2*ones(1,2);

ispeed = ones(tam);
ispeed2 = ones(tam, tam, 2);

gt = zeros(tam);

for i=1:tam
    for j=1:tam
        diff = h.*([i, j] - initials);
        gt(i,j) = sqrt(diff*diff.');
        ispeed2(i,j,1) = diff(1)/gt(i,j);
        ispeed2(i,j,2) = diff(2)/gt(i,j);
        ispeed(i,j) = sqrt(ispeed2(i,j,1)*ispeed2(i,j,1) + ispeed2(i,j,2)*ispeed2(i,j,2));
    end
end

it = 1;
DispResults;