
interp = {'linear'; 'quad'; 'spline'; 'hermite'; 'pchip'};
mode = {'gradient'; 'hopf_lax'; 'golden_search'};


cont = 1;
for j=1:size(interp,1)
    for k=1:size(mode,1)
        label = ['wmm_', interp{j}, '_', mode{k}];
        disp(label);
        t = cputime;
        for i=1:it
            surf{cont} = wmm2D(ispeed2, initials, h, interp{j}, mode{k});
        end
        timec(cont) = (cputime - t)/it;
        cont = cont + 1;
    end
end


format short;
cont = 1;
for j=1:size(interp,1)
    for k=1:size(mode,1)
        label = ['wmm_', interp{j}, '_', mode{k}];
        diff = abs(gt - surf{cont});
        diffv = diff(:);
        l_1 = sum(diffv)/tam.^2;
        l_2 = (diffv.'*diffv)/tam.^2;
        l_inf = max(diffv);
        stringv = [label ' -> l_1 = ' num2str(l_1) ', l_2 = ' num2str(l_2) ', l_inf = ' num2str(l_inf) ', time = ' num2str(timec(cont)) 's'];
        disp(stringv);
        cont = cont + 1;
    end
end


