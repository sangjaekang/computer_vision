im_path = 'image\out_of_focus0015.jpg';
im = rgb2gray(im2double(imread(im_path)));

[im_height, im_width] = size(im);
offset = (patchsize - 1)/2;
patchsize = 11;

% dx = [diff(im, 1, 2), -im(:,end,:)]
% dy = [diff(im, 1, 1); -im(end,:,:)]

dx = [diff(im, 1, 2), -im(:,end,:)];
dy = [diff(im, 1, 1); -im(end,:,:)];

%Rearrange image to patches
dx_col = im2col(dx, [patchsize, patchsize]);
dx_col = bsxfun(@rdivide, dx_col, sum(dx_col));
dy_col = im2col(dy, [patchsize, patchsize]);
dy_col = bsxfun(@rdivide, dy_col, sum(dy_col));

% Compute Kurtosis
normXsquare = bsxfun(@minus, dx_col, mean(dx_col)).^2;
normYsquare = bsxfun(@minus, dy_col, mean(dx_col)).^2;

a = mean(normXsquare.^2);
b =(mean(normXsquare).^2);

qx = mean(normXsquare.^2)./(mean(normXsquare).^2);
qy = mean(normYsquare.^2)./(mean(normYsquare).^2);

qx = reshape(qx, [im_height-patchsize+1, im_width-patchsize+1]);
qy = reshape(qy, [im_height-patchsize+1, im_width-patchsize+1]);

% Normalize for output
qx = log(padarray(qx, [offset, offset], 'replicate'));
qy = log(padarray(qy, [offset, offset], 'replicate'));

qx(isnan(qx)) = min(min(qx(~isnan(qx))));
qy(isnan(qy)) = min(min(qy(~isnan(qy))));

q  = min(qx, qy);


