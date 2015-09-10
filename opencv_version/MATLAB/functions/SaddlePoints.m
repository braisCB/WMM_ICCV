function [ sp ] = SaddlePoints( U, gU, I )
%SADDLEPOINTS Summary of this function goes here
%   Detailed explanation goes here

sp = zeros(100000, 11);
cont = 0;

for i=2:size(U,1)-1
    for j=2:size(U,2)-1
        signo(1) = sign(U(i-1,j-1) - U(i, j));
        signo(2) = sign(U(i-1,j) - U(i, j));
        signo(3) = sign(U(i-1,j+1) - U(i, j));
        signo(4) = sign(U(i,j+1) - U(i, j));
        signo(5) = sign(U(i+1,j+1) - U(i, j));
        signo(6) = sign(U(i+1,j) - U(i, j));
        signo(7) = sign(U(i+1,j-1) - U(i, j));
        signo(8) = sign(U(i,j-1) - U(i, j));
        
        nc = signo(8)*signo(1) <= 0;
        for t=2:8
            nc = nc + (signo(t-1)*signo(t) <= 0);
        end
        if nc == 4 && gU(i,j) > 200
            cont = cont + 1;
            if signo(2) < 0 && signo(6) < 0
                sp(cont, :) = [U(i,j), i, j, i-1, j, i+1, j, i, j-1, i, j+1];
            else
                sp(cont, :) = [U(i,j), i, j, i, j-1, i, j+1, i-1, j, i+1, j];
            end
        end 
%         if U(max(1, i-1), j) < U(i,j) && U(min(size(U,1), i+1), j) < U(i,j)
%             cont = cont + 1;
%             sp(cont, :) = [U(i,j), i, j, i-1, j, i+1, j];
%         elseif (U(i, max(1, j-1)) < U(i,j) && U(i, min(size(U,2), j+1)) < U(i,j))
%             cont = cont + 1;
%             sp(cont, :) = [U(i,j), i, j, i, j-1, i, j+1];
%         end
    end
end

sp = sp(1:cont, :);

end

