clc
clear all
close all

addpath(genpath('..'));

h = [1, 1];
tam = 101;

initials = [21 24; 65 74];

ispeed = ones(tam);
ispeed2 = ones(tam, tam, 2);

gt = Inf*ones(tam);

for p=1:size(initials, 1)
    for i=1:tam
        for j=1:tam
            diff = h.*([i, j] - initials(p,:));
            nval = sqrt(diff*diff.');
            if nval < gt(i,j)
                ispeed2(i,j,1) = diff(1)/nval;
                ispeed2(i,j,2) = diff(2)/nval;
                gt(i,j) = nval;
            end
        end
    end
end
for p=1:size(initials, 1)
    i = initials(p,1);
    j = initials(p,2);
    ispeed2(i,j,1) = 1;
    ispeed2(i,j,2) = 0;
end

it = 1;
DispResults;
