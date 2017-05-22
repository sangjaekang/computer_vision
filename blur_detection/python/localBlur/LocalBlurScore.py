def LocalBlurScore(im, patchsize):
    im_height, im_width, im_color = im.shape
    offset = (patchsize - 1)/2
    x_start = offset + 1; x_end = im_width - offset
    y_start = offset + 1; y_end = im_height - offset
    datasize = (x_end-x_start+1) * (y_end-y_start+1)

    # Feature Extraction
    q1 = LocalKurtosis(im, patchsize)
    q2 = GradientHistogramSpan(im, patchsize)
    q3 = LocalPowerSpectrumSlope(im, patchsize)
