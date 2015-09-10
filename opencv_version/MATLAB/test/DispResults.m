labels = {'msfm1     ';'msfm2     ';'wmmLINGrad';'wmmLINHL  ';'wmmLINGS  ';'wmmSPLGrad';'wmmSPLHL  ';'wmmSPLGS  ';
            'wmmHERGrad';'wmmHERHL  ';'wmmHERGS  ';'wmmPCHGrad';'wmmPCHHL  ';'wmmPCHGS  ';'wmmQUAGrad';'wmmQUAHL  ';'wmmQUAGS  ';
            'fmm1      ';'fmm2      '};
        
disp(labels{1});
t = cputime;
for i=1:it
    surf{1} = msfm2D(ispeed, initials, h, 1);
end
timec(1) = (cputime - t)/it;
disp(labels{2});
t = cputime;
for i=1:it
    surf{2} = msfm2D(ispeed, initials, h, 2);
end
timec(2) = (cputime - t)/it;
disp(labels{3});
t = cputime;
for i=1:it
    surf{3} = wmm2D(ispeed2, initials, h, 'gradient', 'linear');
end
timec(3) = (cputime - t)/it;
disp(labels{4});
t = cputime;
for i=1:it
    surf{4} = wmm2D(ispeed2, initials, h, 'hopf_lax', 'linear');
end 
timec(4) = (cputime - t)/it;
disp(labels{5});
t = cputime;
for i=1:it
    surf{5} = wmm2D(ispeed2, initials, h, 'golden_search', 'linear');
end
timec(5) = (cputime - t)/it;
disp(labels{6});
t = cputime;
for i=1:it
    surf{6} = wmm2D(ispeed2, initials, h, 'gradient', 'spline');
end
timec(6) = (cputime - t)/it;
disp(labels{7});
t = cputime;
for i=1:it
    surf{7} = wmm2D(ispeed2, initials, h, 'hopf_lax', 'spline');
end
timec(7) = (cputime - t)/it;
disp(labels{8});
t = cputime;
for i=1:it
    surf{8} = wmm2D(ispeed2, initials, h, 'golden_search', 'spline');
end
timec(8) = (cputime - t)/it;
disp(labels{9});
t = cputime;
for i=1:it
    surf{9} = wmm2D(ispeed2, initials, h, 'gradient', 'hermite');
end
timec(9) = (cputime - t)/it;
disp(labels{10});
t = cputime;
for i=1:it
    surf{10} = wmm2D(ispeed2, initials, h, 'hopf_lax', 'hermite');
end
timec(10) = (cputime - t)/it;
disp(labels{11});
t = cputime;
for i=1:it
    surf{11} = wmm2D(ispeed2, initials, h, 'golden_search', 'hermite');
end
timec(11) = (cputime - t)/it;
disp(labels{12});
t = cputime;
for i=1:it
    surf{12} = wmm2D(ispeed2, initials, h, 'gradient', 'pchip');
end
timec(12) = (cputime - t)/it;
disp(labels{13});
t = cputime;
for i=1:it
    surf{13} = wmm2D(ispeed2, initials, h, 'hopf_lax', 'pchip');
end
timec(13) = (cputime - t)/it;
disp(labels{14});
t = cputime;
for i=1:it
    surf{14} = wmm2D(ispeed2, initials, h, 'golden_search', 'pchip');
end
timec(14) = (cputime - t)/it;
disp(labels{15});
t = cputime;
for i=1:it
    surf{15} = wmm2D(ispeed2, initials, h, 'gradient', 'quadric');
end
timec(15) = (cputime - t)/it;
disp(labels{16});
t = cputime;
for i=1:it
    surf{16} = wmm2D(ispeed2, initials, h, 'hopf_lax', 'quadric');
end
timec(16) = (cputime - t)/it;
disp(labels{17});
t = cputime;
for i=1:it
    surf{17} = wmm2D(ispeed2, initials, h, 'golden_search', 'quadric');
end
timec(17) = (cputime - t)/it;
disp(labels{18});
t = cputime;
for i=1:it
    surf{18} = fmm2D(ispeed, initials, h, 1);
end
timec(18) = (cputime - t)/it;
disp(labels{19});
t = cputime;
for i=1:it
    surf{19} = fmm2D(ispeed, initials, h, 2);
end
timec(19) = (cputime - t)/it;

format short;
for i=1:19
    diff = abs(gt - surf{i});
    diffv = diff(:);
    l_1 = sum(diffv)/tam.^2;
    l_2 = (diffv.'*diffv)/tam.^2;
    l_inf = max(diffv);
    stringv = [labels{i} ' -> l_1 = ' num2str(l_1) ', l_2 = ' num2str(l_2) ', l_inf = ' num2str(l_inf) ', time = ' num2str(timec(i)) 's'];
    disp(stringv);
end


