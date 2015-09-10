clc
clear all
close all

addpath(genpath('..'));

h = [1, 1];
tam = 50;

initials = [26 26];

ispeed2 = zeros(tam, tam, 2);
ispeed = zeros(tam);
gt = Inf(tam);

for p=1:size(initials, 1)
    for i=1:tam
        for j=1:tam
            diff = h.*([i, j] - initials(p,:));
            nval = 1 - cos(diff(1)/20)*cos(diff(2)/10);
            if nval < gt(i,j)
                dy = 1/20*sin(diff(1)/20)*cos(diff(2)/10);
                dx = 1/10*cos(diff(1)/20)*sin(diff(2)/10);
                ispeed(i,j) = 1/sqrt(dx*dx + dy*dy);
                ispeed2(i,j,1) = dy;
                ispeed2(i,j,2) = dx;
                gt(i,j) = nval;
            end
        end
    end
end

it = 1;
DispResults;