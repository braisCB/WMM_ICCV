function [ e ] = ChanVeseEnergy( U, I, init, sI)
%SADDLEPOINTS Summary of this function goes here
%   Detailed explanation goes here

v = ones(size(U));

v(sI(2), sI(3)) = 0;
p1 = BackPropHeun(U, [1, 1], init, [sI(4), sI(5)]);
p2 = BackPropHeun(U, [1, 1], init, [sI(6), sI(7)]);

for i=1:size(p1,1)
    v(p1(i,1), p1(i,2)) = 0;
end
for i=1:size(p2,1)
    v(p2(i,1), p2(i,2)) = 0;
end

L = bwlabel(v, 4);

nB = max(L(:));
if nB < 2
    e = Inf;
else
    lb = zeros(nB, 2);
    for i=1:nB
        lb(i,2) = i;
        lb(i,1) = sum(L(:) == i);
    end
    lb = sortrows(lb, 1);
    if lb(1,1) < 500 || mean(I(L == lb(1,2))) < 100
        e = Inf;
    else
        labels = lb(1:2, 2);
        e = var(I(L == labels(1)), 1) + var(I(L == labels(2)), 1);
    end
end

end

