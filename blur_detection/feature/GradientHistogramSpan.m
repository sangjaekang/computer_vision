% GradientHistgramSpan -- To measure the heavytailedness of a distribution
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
function [q]=GradientHistogramSpan(im, patchsize)
dx = [diff(im, 1, 2), -im(:,end,:)];
dy = [diff(im, 1, 1); -im(end,:,:)];
mag = sqrt(dx.^2 + dy.^2);

[im_height, im_width] = size(im);
offset = (patchsize-1)/2;

% Gradient histogram span
mag_col = im2col(mag, [patchsize, patchsize]);
mag_col_dup = [mag_col; -mag_col];

num = size(mag_col, 2);
sigma1 = size(1, num);
parfor i = 1 : num
    [V1, V2] = EM_GMM_2(mag_col_dup(:,i));
    s0 = sqrt(V1);
    s1 = sqrt(V2);
    if s0 > s1
        temp = s0;  s0 = s1;  s1 = temp;    
    end;
    sigma1(i) = s1;
end
q = reshape(sigma1, [im_height-patchsize+1, im_width-patchsize+1]);
q = q(4:im_height-patchsize+1-3, 4:im_width-patchsize+1-3);
q = padarray(q, [offset+3, offset+3], 'replicate');


