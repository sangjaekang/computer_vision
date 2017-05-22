% localBlurScore -- Extract blur feature for a fixed scale
%
%   Paras:
%   @im        : Input grayscale image.
%   @patchsize : Parameter controlling local patch size.
%   @W         : Pretrained weight for learned filter.
%   @nb        : Naive Bayesian classifier
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
function feature = localBlurScore(im, patchsize, W, nb)

[im_height, im_width] = size(im);
offset = (patchsize - 1)/2;
x_start = offset+1;  x_end = im_width-offset;
y_start = offset+1;  y_end = im_height-offset;
datasize = (x_end-x_start+1)*(y_end-y_start+1);

% Feature Extraction
q1 = LocalKurtosis(im, patchsize);
q2 = GradientHistogramSpan(im, patchsize);
q3 = LocalPowerSpectrumSlope(im, patchsize);
q4 = LocalLearnedFilter(im, W(:,1:5));

data = zeros(datasize,5);
data(:,1) = reshape(q1(y_start:y_end,x_start:x_end),datasize,1);
data(:,2) = reshape(q2(y_start:y_end,x_start:x_end),datasize,1);
data(:,3) = reshape(q3(y_start:y_end,x_start:x_end),datasize,1);
data(:,4) = reshape(q4{1}(y_start:y_end,x_start:x_end),datasize,1);
data(:,5) = reshape(q4{2}(y_start:y_end,x_start:x_end),datasize,1);

for i = 1:5
    idx = isnan(data(:,i));
    if (sum(idx(:)) == im_height* im_width)
        data(:,i) = 0;
    elseif (sum(idx)>0)
        data(idx,i) = min(data(~idx, i));
    end
end

% Learn blur score from NB classifier
post = posterior(nb,data);
post = mat2gray(post(:,2));

feature = zeros(im_height, im_width);
feature(y_start:y_end,x_start:x_end) = reshape(post,y_end-y_start+1,x_end-x_start+1);


