clc
clear all
close all

addpath(genpath('..'));

h = [1/20, 1/20];
tam = 21;

initials = [1, 1];

ispeed = zeros(tam);
ispeed2 = zeros(tam, tam, 2);

gt = zeros(tam);

for i=1:tam
    for j=1:tam
        diff = h.*([i, j] - initials);
        gt(i,j) = diff(1)*diff(1)/0.05.^2 + diff(2)*diff(2)/0.08.^2;
        ispeed(i,j) = 1/(sqrt(diff(1)*diff(1)/100 + diff(2)*diff(2)/2500));
        ispeed2(i,j,1) = 2*diff(1)/0.05.^2;
        ispeed2(i,j,2) = 2*diff(2)/0.08.^2;
        ispeed(i,j) = 1/sqrt(ispeed2(i,j,1).^2 + ispeed2(i,j,2).^2);
    end
end
%ispeed(isinf(ispeed)) = 50;

it = 1;
DispResults;