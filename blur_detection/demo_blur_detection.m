% demo_blur_detection -- demo for blur detection in for a given input
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
clear;

% Change the following parallel computing setting according to your environment.
%parpool open 4;

addpath('feature');
addpath('UGM');

im_path = 'image\out_of_focus0015.jpg';
im = rgb2gray(im2double(imread(im_path)));

final_map = blurDetection(im);

figure,imshow(final_map);