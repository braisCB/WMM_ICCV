function [ pimage ] = PaintPath( image, path, color )

pimage = image;

for i=1:size(path,1)
    for j=1:length(color)
        pimage(path(i,1), path(i,2), j) = color(j);
    end
end

end