clc
clear all
close all

addpath(genpath('..'));

brain = double(imread('brain.jpg'));

[s1, s2] = size(brain);

brain3 = zeros(s1, s2, 3);
brain3(:,:,1) = brain;
brain3(:,:,2) = brain;
brain3(:,:,3) = brain;

p1 = [270, 188];
p2 = [220, 211];

%% SPEED OF MOTION

speed2_1 = zeros(s1, s2, 2);
speed2_2 = zeros(s1, s2, 2);
ispeed = zeros(s1, s2);

for i=1:s1
    for j=1:s2
        diff_1 = [i,j] - p1;
        diff_2 = [i,j] - p2;
        if i > 1 && i < s1
            dx = (brain(i+1, j) - brain(i-1, j))/2;
        elseif i > 1
            dx = (brain(i, j) - brain(i-1, j));
        elseif i < s1
            dx = (brain(i+1, j) - brain(i, j))/2;
        else
            dx = 0;
        end
        if j > 1 && j < s2
            dy = (brain(i, j+1) - brain(i, j-1))/2;
        elseif j > 1
            dy = (brain(i, j) - brain(i, j-1));
        elseif j < s2
            dy = (brain(i, j+1) - brain(i, j))/2;
        else
            dy = 0;
        end
        ny1 = diff_1(1) + dy;
        nx1 = diff_1(2) + dx;
        ny2 = diff_2(1) + dy;
        nx2 = diff_2(2) + dx;
        dy1 = ny1/sqrt(ny1*ny1 + nx1*nx1)/(1 + dx*dx + dy*dy);
        dx1 = nx1/sqrt(ny1*ny1 + nx1*nx1)/(1 + dx*dx + dy*dy);
        dy2 = ny2/sqrt(ny2*ny2 + nx2*nx2)/(1 + dx*dx + dy*dy);
        dx2 = nx2/sqrt(ny2*ny2 + nx2*nx2)/(1 + dx*dx + dy*dy);
        speed2_1(i,j,1) = dy1;
        speed2_1(i,j,2) = dx1;
        speed2_2(i,j,1) = dy2;
        speed2_2(i,j,2) = dx2;
        ispeed(i,j) = (dx*dx + dy*dy + 1);
    end
end


%% WMM GRADIENT SPLINE

surf1 = wmm2D(speed2_1, p1, [1 1], 'gradient', 'spline');
surf11 = wmm2D(speed2_1, p2, [1 1], 'gradient', 'spline');

sp1 = SaddlePoints(surf1, ispeed, brain);
sp11 = SaddlePoints(surf11, ispeed, brain);

ewinner = [Inf 0];

for i=1:size(sp1, 1)
    disp(['sp1: ' num2str(i) ' >-< ' num2str(size(sp1, 1))]);
    e = ChanVeseEnergy( surf1, brain, p1, sp1(i,:));
    if e < ewinner(1)
        ewinner = [e i];
    end
end

ewinner1 = [Inf 0];
for i=1:size(sp11, 1)
    disp(['sp11: ' num2str(i) ' >-< ' num2str(size(sp11, 1))]);
    e = ChanVeseEnergy( surf11, brain, p2, sp11(i,:));
    if e < ewinner1(1)
        ewinner1 = [e i];
    end
end

path1 = BackPropHeun(surf1, [1, 1], p1, [sp1(ewinner(2), 4), sp1(ewinner(2), 5)]);
path12 = BackPropHeun(surf1, [1, 1], p1, [sp1(ewinner(2), 6), sp1(ewinner(2), 7)]);
brain31 = brain3;
brain31 = PaintPath(brain31, [sp1(ewinner(2), 2), sp1(ewinner(2), 3); path1; path12], [255, 0, 0]);

path11 = BackPropHeun(surf11, [1, 1], p2, [sp11(ewinner1(2), 4), sp11(ewinner1(2), 5)]);
path112 = BackPropHeun(surf11, [1, 1], p2, [sp11(ewinner1(2), 6), sp11(ewinner1(2), 7)]);
brain31 = PaintPath(brain31, [sp11(ewinner1(2), 2), sp11(ewinner1(2), 3); path11; path112], [0, 255, 0]);

figure();
imagesc(uint8(brain31));
hold on
plot(p1(2), p1(1), 'xg','MarkerSize',20,'LineWidth',5);
plot(p2(2), p2(1), 'xr','MarkerSize',20,'LineWidth',5);
hold off
axis off;
axis([145 230 200 280]);

print -depsc brain_wmmgradspl.eps

%% WMM MODIFIED HOPF-LAX SPLINE

surf2 = wmm2D(speed2_1, p1, [1 1], 'hopf_lax', 'spline');
surf21 = wmm2D(speed2_1, p2, [1 1], 'hopf_lax', 'spline');

sp2 = SaddlePoints(surf2, ispeed, brain);
sp21 = SaddlePoints(surf21, ispeed, brain);

ewinner = [Inf 0];

for i=1:size(sp2, 1)
    disp(['sp2: ' num2str(i) ' >-< ' num2str(size(sp2, 1))]);
    e = ChanVeseEnergy( surf2, brain, p1, sp2(i,:));
    if e < ewinner(1)
        ewinner = [e i];
    end
end

ewinner1 = [Inf 0];
for i=1:size(sp21, 1)
    disp(['sp21: ' num2str(i) ' >-< ' num2str(size(sp21, 1))]);
    e = ChanVeseEnergy( surf21, brain, p2, sp21(i,:));
    if e < ewinner1(1)
        ewinner1 = [e i];
    end
end

path2 = BackPropHeun(surf2, [1, 1], p1, [sp2(ewinner(2), 4), sp2(ewinner(2), 5)]);
path22 = BackPropHeun(surf2, [1, 1], p1, [sp2(ewinner(2), 6), sp2(ewinner(2), 7)]);
brain32 = brain3;
brain32 = PaintPath(brain32, [sp2(ewinner(2), 2), sp2(ewinner(2), 3); path2; path22], [255, 0, 0]);

path21 = BackPropHeun(surf21, [1, 1], p2, [sp21(ewinner1(2), 4), sp21(ewinner1(2), 5)]);
path212 = BackPropHeun(surf21, [1, 1], p2, [sp21(ewinner1(2), 6), sp21(ewinner1(2), 7)]);
brain32 = PaintPath(brain32, [sp21(ewinner1(2), 2), sp21(ewinner1(2), 3); path21; path212], [0, 255, 0]);

figure();
imagesc(uint8(brain32));
hold on
plot(p1(2), p1(1), 'xg','MarkerSize',20,'LineWidth',5);
plot(p2(2), p2(1), 'xr','MarkerSize',20,'LineWidth',5);
hold off
axis off;
axis([145 230 200 280]);

print -depsc brain_wmmhlspl.eps

%% FFM 2nd ORDER

surf3 = fmm2D(ispeed, p1, [1 1], 2);
surf31 = fmm2D(ispeed, p2, [1 1], 2);

sp3 = SaddlePoints(surf3, ispeed, brain);
sp31 = SaddlePoints(surf31, ispeed, brain);

ewinner = [Inf 0];

for i=1:size(sp3, 1)
    disp(['sp3: ' num2str(i) ' >-< ' num2str(size(sp31, 1))]);
    e = ChanVeseEnergy( surf3, brain, p1, sp3(i,:));
    if e < ewinner(1)
        ewinner = [e i];
    end
end

ewinner1 = [Inf 0];
for i=1:size(sp31, 1)
    disp(['sp31: ' num2str(i) ' >-< ' num2str(size(sp31, 1))]);
    e = ChanVeseEnergy( surf31, brain, p2, sp31(i,:));
    if e < ewinner1(1)
        ewinner1 = [e i];
    end
end

path3 = BackPropHeun(surf3, [1, 1], p1, [sp3(ewinner(2), 4), sp3(ewinner(2), 5)]);
path32 = BackPropHeun(surf3, [1, 1], p1, [sp3(ewinner(2), 6), sp3(ewinner(2), 7)]);
brain33 = brain3;
brain33 = PaintPath(brain33, [sp3(ewinner(2), 2), sp3(ewinner(2), 3); path3; path32], [255, 0, 0]);

path31 = BackPropHeun(surf31, [1, 1], p2, [sp31(ewinner1(2), 4), sp31(ewinner1(2), 5)]);
path312 = BackPropHeun(surf31, [1, 1], p2, [sp31(ewinner1(2), 6), sp31(ewinner1(2), 7)]);
brain33 = PaintPath(brain33, [sp31(ewinner1(2), 2), sp31(ewinner1(2), 3); path31; path312], [0, 255, 0]);

figure();
imagesc(uint8(brain33));
hold on
plot(p1(2), p1(1), 'xg','MarkerSize',20,'LineWidth',5);
plot(p2(2), p2(1), 'xr','MarkerSize',20,'LineWidth',5);
hold off
axis off;
axis([145 230 200 280]);

print -depsc brain_fmm2.eps

%% MSFM 2nd ORDER

surf4 = msfm2D(ispeed, p1, [1 1], 2);
surf41 = msfm2D(ispeed, p2, [1 1], 2);

sp4 = SaddlePoints(surf4, ispeed, brain);
sp41 = SaddlePoints(surf41, ispeed, brain);

ewinner = [Inf 0];

for i=1:size(sp4, 1)
    disp(['sp4: ' num2str(i) ' >-< ' num2str(size(sp41, 1))]);
    e = ChanVeseEnergy( surf4, brain, p1, sp4(i,:));
    if e < ewinner(1)
        ewinner = [e i];
    end
end

ewinner1 = [Inf 0];
for i=1:size(sp41, 1)
    disp(['sp41: ' num2str(i) ' >-< ' num2str(size(sp41, 1))]);
    e = ChanVeseEnergy( surf41, brain, p2, sp41(i,:));
    if e < ewinner1(1)
        ewinner1 = [e i];
    end
end

path4 = BackPropHeun(surf4, [1, 1], p1, [sp4(ewinner(2), 4), sp4(ewinner(2), 5)]);
path42 = BackPropHeun(surf4, [1, 1], p1, [sp4(ewinner(2), 6), sp4(ewinner(2), 7)]);
brain34 = brain3;
brain34 = PaintPath(brain34, [sp4(ewinner(2), 2), sp4(ewinner(2), 3); path4; path42], [255, 0, 0]);

path41 = BackPropHeun(surf41, [1, 1], p2, [sp41(ewinner1(2), 4), sp41(ewinner1(2), 5)]);
path412 = BackPropHeun(surf41, [1, 1], p2, [sp41(ewinner1(2), 6), sp41(ewinner1(2), 7)]);
brain34 = PaintPath(brain34, [sp41(ewinner1(2), 2), sp41(ewinner1(2), 3); path41; path412], [0, 255, 0]);

figure();
imagesc(uint8(brain34));
hold on
plot(p1(2), p1(1), 'xg','MarkerSize',20,'LineWidth',5);
plot(p2(2), p2(1), 'xr','MarkerSize',20,'LineWidth',5);
hold off
axis off;
axis([145 230 200 280]);

print -depsc brain_msfm2.eps

