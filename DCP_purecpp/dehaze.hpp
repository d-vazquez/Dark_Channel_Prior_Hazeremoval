//
//  dehaze.hpp
//  Dark_Channel_Prior_Hazeremoval
//
//  Created by Dario Vazquez on 2/13/23.
//

#ifndef dehaze_hpp
#define dehaze_hpp

#include <stdio.h>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "Image.hpp"

#ifndef MIN
#  define MIN(a,b)  ((a) > (b) ? (b) : (a))
#endif

#ifndef MAX
#  define MAX(a,b)  ((a) < (b) ? (b) : (a))
#endif

// Optimize for 2D kernels
#define CONV_KERNEL1D

template<typename T>
std::vector<size_t> argsort(const std::vector<T> &array);

void imshow_float_array(char *name, Image& _im);
void DarkChannel(Image& im, Image& dark, int sz);
void erode(Image& src, Image& kernel, Image& dst);
void AtmLight(Image& img, Image& dark, float* airlight);
void TransmissionEstimate(Image& im, float* airlight, Image& dst, int sz);
void TransmissionRefine(Image& im, Image& et, Image& dst);
void rgb2grey(Image& src, Image& dst);
void Guidedfilter(Image& im, Image& p, int r, float eps, Image& dst);
void boxFilter(Image& src, Image& dst, int width, int height);
void convoluteKernel(Image& src, Image& kernel, Image& dst);
void Recover(Image &im, Image &t, Image& dst, float* A, float tx);
float erode_operation(float *image_pixel, float *kernel_pixel, float *accumulator);

#endif /* dehaze_hpp */
