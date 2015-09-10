clc
clear all
close all

addpath(genpath('..'));

h = [.2, .1];
tam = 101;

initials = (tam+1)/2*ones(1,2);

ispeed = zeros(tam);
ispeed2 = zeros(tam, tam, 2);

gt = zeros(tam);

for i=1:tam
    for j=1:tam
        diff = h.*([i, j] - initials);
        gt(i,j) = diff(1)*diff(1)/20 + diff(2)*diff(2)/100;
        ispeed(i,j) = 1/(sqrt(diff(1)*diff(1)/100 + diff(2)*diff(2)/2500));
        ispeed2(i,j,1) = 2*diff(1)/20;
        ispeed2(i,j,2) = 2*diff(2)/100;
    end
end
%ispeed(isinf(ispeed)) = 50;

it = 1;
DispResults;