    % blurDetection -- Blur detection in for a given input
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

function final_map = blurDetection(im)

s1 = 11;        % patch size for scale 1
s2 = 15;        % patch size for scale 2
s3 = 21;        % patch size for scale 3
alpha = 0.5;    % weight for multiscale inference

load('nb_classifier.mat');
load('learned_linear_filter.mat');

fprintf('Extracting level 1 feature...\n');
feature.scale1 = localBlurScore(im, s1, W_11, nb11);
fprintf('Extracting level 2 feature...\n');
feature.scale2 = localBlurScore(im, s2, W_15, nb15);
fprintf('Extracting level 3 feature...\n');
feature.scale3 = localBlurScore(im, s3, W_21, nb21);

fprintf('Multiscale Inference...\n');
final_map = multiScaleBlurInference(feature, alpha, s1, s2, s3);
