function [ path ] = BackPropHeun( U, h, pinit, pend )

[s1, s2] = size(U);

target = pend;
itarget = target;

path = zeros(1000,2);
path(1,:) = pend;
cont = 1;

while sum(target == pinit) < length(pinit)
    diffy = [- U(target(1), target(2)) + U(max(1, target(1) - 1), target(2)), ...
        U(target(1), target(2)) - U(min(size(U,1), target(1) + 1), target(2))];
    diffx = [- U(target(1), target(2)) + U(target(1), max(1, target(2) - 1)), ...
        U(target(1), target(2)) - U(target(1), min(target(2) + 1, size(U,2)))];
    if diffx(1) > 0
        diffx(1) = 0;
    end
    if diffx(2) < 0
        diffx(2) = 0;
    end
    if diffy(1) > 0
        diffy(1) = 0;
    end
    if diffy(2) < 0
        diffy(2) = 0;
    end
    if -diffx(1) > diffx(2)
        dx = diffx(1);
    else
        dx = diffx(2);
    end
    if -diffy(1) > diffy(2)
        dy = diffy(1);
    else
        dy = diffy(2);
    end
    if dx ~= 0 && dy ~= 0
        factor = 1/max(abs([dx dy]))*[dy dx];
        itarget = itarget + factor;
        if sum(target == round(itarget)) < length(target)
            cont = cont + 1;
            path(cont, :) = round(itarget);
        end
        target = round(itarget);
    else
        diff = 0;
        ntarget = target;
        for i=-1:1
            for j=-1:1
                pdiff = h.*([i, j] - pinit);
                dist = sqrt(pdiff*pdiff.');
                newpos = [i, j] + target;
                valid = newpos(1) > 0 && newpos(1) <= s1 && newpos(2) > 0 && newpos(2) <= s2;
                if valid
                    ndiff = (U(target(1), target(2)) - U(newpos(1), newpos(2)))/dist;
                    if ndiff > diff
                        ntarget = newpos;
                        diff = ndiff;
                    end
                end
            end
        end
        if sum(target == ntarget) < length(target)
            cont = cont + 1;
            path(cont, :) = ntarget;
        end
        target = ntarget;
        itarget = target;
    end
end

path = path(1:cont, :);

end