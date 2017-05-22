% multiScaleBlurInference -- Blur inference given feature map from multiple 
%   scale. This function solves Eq.(21) in the paper.
%
%   Paras: 
%   @im        : Input grayscale image.
%   @alpha     : Parameter controlling weight for multiscale inference.
%   @s1        : Patch size for scale 1.                       
%   @s2        : Patch size for scale 2.
%   @s3        : Patch size for scale 3.
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
function [final_fea_map] = multiScaleBlurInference(im, alpha, s1, s2, s3)
%% Parameters
nStates = 2;
[nRows, nCols] = size(im.scale1);

%% Graphic model generation
[adj, nNodes] = loadAdjMatrix(nRows, nCols, s3, s2, s1);

edgeStruct = UGM_makeEdgeStruct(adj,nStates);
offset1 = (s1-1)/2; offset2 = (s2-1)/2; offset3 = (s3-1)/2;
fea_scale1 = im.scale1(offset1+1:nRows-offset1,offset1+1:nCols-offset1);
fea_scale2 = im.scale2(offset2+1:nRows-offset2,offset2+1:nCols-offset2);
fea_scale3 = im.scale3(offset3+1:nRows-offset3,offset3+1:nCols-offset3);

X = [fea_scale3(:); fea_scale2(:); fea_scale1(:)];
Xstd = UGM_standardizeCols(reshape(X,[1 1 nNodes]),1);

% Make nodePot
nodePot = zeros(nNodes,nStates);
nodePot(:,1) = exp(-abs(1-Xstd(:)));
nodePot(:,2) = exp(-abs(Xstd(:)));

% Make edgePot
edgePot = zeros(nStates,nStates,edgeStruct.nEdges);
for e = 1:edgeStruct.nEdges
    pot_same = 1;
    pot_diff = exp(-alpha);
    edgePot(:,:,e) = [pot_same pot_diff; pot_diff pot_same];
end

%% Hierachical inference
[nodeBel, ~, ~] = UGM_Infer_LBP(nodePot,edgePot,edgeStruct);

%% Seperate to three maps
[nRows3, nCols3] = computeSize(nRows, nCols, s3);
nNodes3 = nRows3*nCols3;

final_fea_map = padarray(reshape(nodeBel(1:nNodes3,2), nRows3, nCols3), ...
    [offset3, offset3], 'replicate');
final_fea_map = final_fea_map * -1 + 1;
end

function [actual_Rows, actual_Cols] = computeSize(nRows, nCols, patchsize)
actual_Rows = nRows - patchsize + 1;
actual_Cols = nCols - patchsize + 1;
end
