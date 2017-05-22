% EM_GMM_2 -- Compute expectation for GMM with two Gaussian Component, where 
% mean are fixed as zero. Our goal is mainly to compute the variance for two 
% Gaussians.
%
%   Paras:
%   @X   : Input data
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
function [V1, V2] = EM_GMM_2(X)

%% Initialize parameters
V1 = 0.5;
V2 = 0.0001;
pi = 0.5;
N = length(X);

pi_init=2*pi;
maxiter = 100;
iter=0;

while (abs(pi_init-pi)/pi_init > 1e-3 && iter < maxiter)
    iter = iter+1;
    pi_init = pi;
    
    XxX = X.*X;
    %% Expectation Step
    pi_x_GPV2 = pi*GaussianProb(XxX,V2);
    gamma = pi_x_GPV2 ./ ((1-pi)*GaussianProb(XxX,V1) + pi_x_GPV2);

    %% Maximization Step
    V1 = (1-gamma)'*(XxX) / sum(1-gamma);
    V2 = gamma'*(XxX) / sum(gamma);
    pi = sum(gamma)/N;
end

%% Function to compute the probability for Gaussian distribution
function [Y] = GaussianProb(XxX, V)
Y = exp(-(XxX)/(2*V))/sqrt(2*pi*V);
