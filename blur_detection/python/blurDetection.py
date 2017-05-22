import localBlur

def blurDetection(im, s1, s2, s3, alpha):

    print("Extracting level 1 feature..\n")
    scale1=localBlur.localBlurScore(im,s1)

    print("Extracting level 2 feature..\n")
    scale2=localBlur.localBlurScore(im,s2)

    print("Extracting level 3 feature..\n")
    scale3=localBlur.localBlurScore(im,s3)
    feature=[scale1, scale2, scale3]

    print("multiscale Inference..\n")
    return localBlur.multiScaleBlurInference(feature,alpha,s1,s2,s3)

