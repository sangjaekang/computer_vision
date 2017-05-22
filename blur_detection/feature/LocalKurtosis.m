% LocalKurtosis -- To measure the peakness of a distribution
%
%   Paras: 
%   @im        : Input grayscale image.
%   @patchsize : Parameter controlling local patch size.
%
%   The Code is created based on the method described in the following paper 
%   [1] "Discriminative Blur Detection Features", Jianping Shi, Li Xu, Jiaya Jia,
%       IEEE Conference on Computer Vision and Pattern Recognition, 2014. 
%   The code and the algorithm are for non-comercial use only.
%  
%   Author: Jianping Shi (jpshi@cse.cuhk.edu.hk)
%   Date  : 03/27/2014
%   Version : 1.0 
%   Copyright 2014, The Chinese University of Hong Kong.
%
function [q] = LocalKurtosis(im, patchsize)

[im_height, im_width] = size(im);
offset = (patchsize - 1)/2;

% Compute image gradient
dx = [diff(im, 1, 2), -im(:,end,:)];
dy = [diff(im, 1, 1); -im(end,:,:)];

% Rearrange image to patches
dx_col = im2col(dx, [patchsize, patchsize]);
dx_col = bsxfun(@rdivide, dx_col, sum(dx_col));
dy_col = im2col(dy, [patchsize, patchsize]);
dy_col = bsxfun(@rdivide, dy_col, sum(dy_col));

% Compute Kurtosis
normXsquare = bsxfun(@minus, dx_col, mean(dx_col)).^2;
normYsquare = bsxfun(@minus, dy_col, mean(dx_col)).^2;

qx = mean(normXsquare.^2)./(mean(normXsquare).^2);
qy = mean(normYsquare.^2)./(mean(normYsquare).^2);

qx = reshape(qx, [im_height-patchsize+1, im_width-patchsize+1]);
qy = reshape(qy, [im_height-patchsize+1, im_width-patchsize+1]);

% Normalize for output
qx = log(padarray(qx, [offset, offset], 'replicate'));
qy = log(padarray(qy, [offset, offset], 'replicate'));

% qx = log(qx);
% qy = log(qy);

qx(isnan(qx)) = min(min(qx(~isnan(qx))));
qy(isnan(qy)) = min(min(qy(~isnan(qy))));

q  = min(qx, qy);
