/* 
 * sf = calculateSf(s, C, f);
 *
 * To Accelerate the following matlab code
 * sf = zeros(size(C));
 * for i = 1:length(C)
 *     sf(i) = sum(s(f==C(i)));
 * end
 */
# include "mex.h"
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){   
    // Macros for the output and input arguments
    #define sf_out  plhs[0]
    #define s_in    prhs[0]
    #define C_in    prhs[1]
    #define f_in    prhs[2]
            
	double *sf, *s, *C, *f;
    int heightC, widthC, lengthC, height, width;
    
    if (nrhs < 3 || nrhs > 4)
        mexErrMsgTxt("Wrong number of input arguments.");
    else if (nlhs > 1)
        mexErrMsgTxt("Too many output arguments.");
    
    height = mxGetM(s_in);
    width = mxGetN(s_in);
    heightC = mxGetM(C_in);
    widthC = mxGetN(C_in);
    lengthC = heightC*widthC;
    
    s = mxGetPr(s_in);
    C = mxGetPr(C_in);
    f = mxGetPr(f_in);
    sf_out = mxCreateDoubleMatrix(lengthC, 1, mxREAL);
    sf = mxGetPr(sf_out);
    
    for (int idxC = 0; idxC < lengthC; idxC++){
        int cnt = 0;
        for (int i = 0; i < height*width; i++){
            if (f[i] == C[idxC]){
                sf[idxC] += s[i];
                cnt ++;
            }
        }
        sf[idxC] /= cnt;
    }
    return;
}