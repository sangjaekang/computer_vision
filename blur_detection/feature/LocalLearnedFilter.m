% LocalLearnedFilter -- To learn blur feature from learned filter
%
%   Paras: 
%   @im        : Input grayscale image.
%   @W         : Learned local filter.
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
function [q] = LocalLearnedFilter(im, W)
[im_height, im_width] = size(im);
[num, filter_num] = size(W);
patchsize = sqrt(num);
offset = (patchsize - 1)/2;

im_col = im2col(im, [patchsize, patchsize]);
features = im_col' * W;
q = cell(filter_num,1);
parfor i = 1 : filter_num
    low = prctile(features(:,i), 3);
    high = prctile(features(:,i), 97);
    q{i} = padarray(reshape(features(:,i), [im_height-patchsize+1, im_width-patchsize+1]),...
        [offset, offset], 'replicate');
    q{i}(q{i}<low) = low;
    q{i}(q{i}>high) = high;
end