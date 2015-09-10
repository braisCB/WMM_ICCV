addpath(genpath('..'));

MEXFLAGS = ' -O -g -L/usr/lib ';

if strcmp(computer(),'GLNXA64') || ...
   strcmp(computer(),'PCWIN64') || ...
   strcmp(computer(),'MACI64')
    MEXFLAGS = [MEXFLAGS,  ' -largeArrayDims'];
end

mexcmd = ['mex ' MEXFLAGS ' c_files/wmm2D_c.cpp'];
disp(mexcmd);
eval(mexcmd);

!mv *.mex* mex/
