% loadAdjMatrix -- Compute adjacency matrix for given input image size
%
%   Paras: 
%   @nRows     : Number of rows for input image.
%   @nCols     : Number of columns for input image.
%   @s1        : Patch size for scale 1.                       
%   @s2        : Patch size for scale 2.
%   @s3        : Patch size for scale 3.
%
%   Note: s1 < s2 < s3
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
function [adj, nNodes] = loadAdjMatrix(nRows, nCols, s1, s2, s3)

[nRows1, nCols1] = computeSize(nRows, nCols, s1);
[nRows2, nCols2] = computeSize(nRows, nCols, s2);
[nRows3, nCols3] = computeSize(nRows, nCols, s3);

nNodes1 = nRows1*nCols1;
nNodes2 = nRows2*nCols2;
nNodes3 = nRows3*nCols3;
nNodes = nNodes1 + nNodes2 + nNodes3;

% adj = sparse(nNodes, nNodes);

y_ind = []; x_ind = [];
%% Add Down edge
% layer 1
ind = 1:nNodes1;
exclude = sub2ind([nRows1 nCols1],repmat(nRows1,[1 nCols1]),1:nCols1);
ind = setdiff(ind,exclude);
y_ind = [y_ind, ind]; x_ind = [x_ind, ind+1];

% layer 2
ind = 1:nNodes2;
exclude = sub2ind([nRows2 nCols2],repmat(nRows2,[1 nCols2]),1:nCols2);
ind = setdiff(ind,exclude);
ind = ind + nNodes1;
y_ind = [y_ind, ind]; x_ind = [x_ind, ind+1];

% layer 3
ind = 1:nNodes3;
exclude = sub2ind([nRows3 nCols3],repmat(nRows3,[1 nCols3]),1:nCols3);
ind = setdiff(ind,exclude);
ind = ind + nNodes1 + nNodes2;
y_ind = [y_ind, ind]; x_ind = [x_ind, ind+1];

%% Add Right Edges
% layer 1
ind = 1:nNodes1;
exclude = sub2ind([nRows1 nCols1],1:nRows1,repmat(nCols1,[1 nRows1]));
ind = setdiff(ind, exclude);
y_ind = [y_ind, ind]; x_ind = [x_ind, ind+nRows1];

% layer 2
ind = 1:nNodes2;
exclude = sub2ind([nRows2 nCols2],1:nRows2,repmat(nCols2,[1 nRows2]));
ind = setdiff(ind, exclude);
ind = ind + nNodes1;
y_ind = [y_ind, ind]; x_ind = [x_ind, ind+nRows2];

% layer 3
ind = 1:nNodes3;
exclude = sub2ind([nRows3 nCols3],1:nRows3,repmat(nCols3,[1 nRows3]));
ind = setdiff(ind, exclude);
ind = ind + nNodes1 + nNodes2;
y_ind = [y_ind, ind]; x_ind = [x_ind, ind+nRows3];

%% Add Link From Layer 1 to Layer 2
y = 1:nRows1; x = 1:nCols1;
[y, x] = meshgrid(y, x);
ind1 = sub2ind([nRows1 nCols1], y, x);
offset = (nRows2 - nRows1) / 2;
y = offset+1:offset+nRows1; x = offset+1:offset+nCols1;
[y, x] = meshgrid(y, x);
ind2 = sub2ind([nRows2 nCols2], y, x) + nNodes1;
y_ind = [y_ind, ind1(:)']; x_ind = [x_ind, ind2(:)'];

%% Add Link From Layer 2 to Layer 3
y = 1:nRows2; x = 1:nCols2;
[y, x] = meshgrid(y, x);
ind1 = sub2ind([nRows2 nCols2], y, x) + nNodes1;
offset = (nRows3 - nRows2) / 2;
y = offset+1:offset+nRows2; x = offset+1:offset+nCols2;
[y, x] = meshgrid(y, x);
ind2 = sub2ind([nRows3 nCols3], y, x) + nNodes1 + nNodes2;
y_ind = [y_ind, ind1(:)']; x_ind = [x_ind, ind2(:)'];

%% Add the reversed links
% adj = adj + adj';
y_ind_final = [y_ind, x_ind]; x_ind_final = [x_ind, y_ind];

%% Generate the sparse Matrix
edgeNum = length(y_ind_final);
edgeValue = ones(1, edgeNum);
adj = sparse(y_ind_final, x_ind_final, edgeValue, nNodes, nNodes);

end

function [actual_Rows, actual_Cols] = computeSize(nRows, nCols, patchsize)
actual_Rows = nRows - patchsize + 1;
actual_Cols = nCols - patchsize + 1;
end