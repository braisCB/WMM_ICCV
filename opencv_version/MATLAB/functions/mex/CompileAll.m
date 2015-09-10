
MEXFLAGS = ' -O -g -L/usr/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -lopencv_flann';

if strcmp(computer(),'GLNXA64') || ...
   strcmp(computer(),'PCWIN64') || ...
   strcmp(computer(),'MACI64')
    MEXFLAGS = [MEXFLAGS,  ' -largeArrayDims'];
end

mexcmd = ['mex ' MEXFLAGS ' -I/usr/include/opencv wmmTurboLinear2D.cpp'];
disp(mexcmd);
eval(mexcmd);
mexcmd = ['mex ' MEXFLAGS ' -I/usr/include/opencv wmmTurboPchip2D.cpp'];
disp(mexcmd);
eval(mexcmd);
mexcmd = ['mex ' MEXFLAGS ' -I/usr/include/opencv wmmTurboSpline2D.cpp'];
disp(mexcmd);
eval(mexcmd);
mexcmd = ['mex ' MEXFLAGS ' -I/usr/include/opencv wmmTurboHermite2D.cpp'];
disp(mexcmd);
eval(mexcmd);
mexcmd = ['mex ' MEXFLAGS ' -I/usr/include/opencv wmmTurboInterp2D.cpp'];
disp(mexcmd);
eval(mexcmd);
mexcmd = ['mex ' MEXFLAGS ' -I/usr/include/opencv msfm2Dc.cpp'];
disp(mexcmd);
eval(mexcmd);
mexcmd = ['mex ' MEXFLAGS ' -I/usr/include/opencv fmm2Dc.cpp'];
disp(mexcmd);
eval(mexcmd);

!mv *.m?* build/

clear all