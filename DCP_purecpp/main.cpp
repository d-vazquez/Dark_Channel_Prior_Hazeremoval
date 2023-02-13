
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"

#include <stdio.h>
#include <iostream>
#include <math.h>
#include <vector>
#include <numeric>

//#include "raw_image.hpp"
#include "Image.hpp"

#ifndef MIN
#  define MIN(a,b)  ((a) > (b) ? (b) : (a))
#endif

#ifndef MAX
#  define MAX(a,b)  ((a) < (b) ? (b) : (a))
#endif

// Optimize for 2D kernels
#define CONV_KERNEL1D

using namespace std::chrono;
using namespace cv;
using namespace std;

template<typename T>
std::vector<size_t> argsort(const std::vector<T> &array);

void imshow_float_array(char *name, Image& _im);
void DarkChannel(Image& im, int sz);
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

void DarkChannel(Image& im, Image& dark, int sz)
{
    Image dc(im.rows(), im.cols(), 1);
    

    for(int i = 0; i < dc.rows(); i++)
    {
        for(int j = 0; j < dc.cols(); j++)
        {
            dc(0,i,j) =
            MIN(im(0,i,j), MIN(im(1,i,j),im(2,i,j)));
        }
    }
   
    
    for(int i = 0; i < dc.rows(); i++)
    {
        for(int j = 0; j < dc.cols(); j++)
        {
            dc(0,i,j) =
            MIN(im(0,i,j), MIN(im(1,i,j),im(2,i,j)));
        }
    }
    
#ifndef CONV_KERNEL1D
    // Create erode kernel
    Image kernel(sz, sz, 1);
    kernel.fill(1.0f);
    
    erode(dc, kernel, dark);
#else
    Image kernel1(1, sz, 1);
    kernel1.fill(-1.0f);
    
    Image kernel2(sz, 1, 1);
    kernel2.fill(-1.0f);
    
    // need an intermediate image
    Image src_dst(dark.rows(), dark.cols(), 1);
    
    erode(dc, kernel1, src_dst);
    erode(src_dst, kernel2, dark);
    
#endif
}

void erode(Image& src, Image& kernel, Image& dst)
{
    int x_delta = floor(kernel.rows()/2);
    int y_delta = floor(kernel.cols()/2);
    
    // Run thru the dst image
    for(int x = 0; x < src.rows(); x++)
    {
        for(int y = 0; y < src.cols(); y++)
        {
            // Convolute the kernel
            
            // we need to keep track of the minimum in the kernel, the max value of
            // the image can be 1.0, so lets start there
            float erode_value = 1.0f;
            
            for(int i = (x - x_delta), i_k = 0;
                i <= (x + x_delta);
                i++, i_k++)
            {
                for(int j = (y - y_delta), j_k = 0;
                    j <= (y + y_delta);
                    j++, j_k++)
                {
                    int i_fixed = i, j_fixed = j;
                    // Check bounds
                    if(i < 0)
                    {
                        // Out of average on the left side, abs(x) is enoug
                        i_fixed = abs(i);
                    }
                    if(j < 0 )
                    {
                        // Out of average on the left side, abs(x) is enoug
                        j_fixed = abs(j);
                    }
                    if(i >= src.rows() )
                    {
                        // Out of average, substract distance from center
                        i_fixed = src.rows() - (i - src.rows() + 1);
                    }
                    if(j >= src.cols())
                    {
                        // Out of average, substract distance from center
                        j_fixed = src.cols() - (j - src.cols() + 1);
                    }
                    
                    erode_value = MIN(erode_value, src(0,i,j));
                }
            }
            
            // kernel has been run thru the image, lets write the MIN value
            dst(0,x,y) = erode_value;
        }
    }
}

void AtmLight(Image& img, Image& dark, float* airlight)
{
    //
    int _rows = img.rows();
    int _cols = img.cols();
    
    int imsz = _rows*_cols;
    
    int numpx = (int)MAX(imsz/1000, 1);
    
    // Copy arrys, need to sort withoout modifying image
    vector<float> darkvec(&img(0,0,0), &img(0,0,0) + dark.rows()*dark.cols());
    
    vector<float> imvec_r(&img(0,0,0), &img(0,0,0) + img.rows()*img.cols());
    vector<float> imvec_g(&img(1,0,0), &img(1,0,0) + img.rows()*img.cols());
    vector<float> imvec_b(&img(2,0,0), &img(2,0,0) + img.rows()*img.cols());
    
    // Get indices of sorted darkvec whcih is the dark flattened matrix
    vector<size_t> indices = argsort(darkvec);
    
    airlight[0] = 0.0f;
    airlight[1] = 0.0f;
    airlight[2] = 0.0f;
    
    for(int ind = imsz-numpx; ind < imsz; ind++)
    {
        airlight[0] += imvec_r[indices[ind]];
        airlight[1] += imvec_g[indices[ind]];
        airlight[2] += imvec_b[indices[ind]];
    }
    
    airlight[0] = airlight[0] / numpx;
    airlight[1] = airlight[1] / numpx;
    airlight[2] = airlight[2] / numpx;
    
}

void  TransmissionEstimate(Image& im, float* airlight, Image& dst, int sz)
{
    float omega = 0.95;
    
    Image im3(im.rows(), im.cols(), im.chan());
    
    for(int x = 0; x < im3.rows(); x++)
    {
        for(int y = 0; y < im3.cols(); y++)
        {
            im3(0,x,y) = im(0,x,y) / airlight[0];
            im3(1,x,y) = im(1,x,y) / airlight[1];
            im3(2,x,y) = im(2,x,y) / airlight[2];
        }
    }
    
    Image dark_temp(im.rows(), im.cols(), 1);
    DarkChannel(im3, dark_temp, sz);
    
    for(int x = 0; x < dst.rows(); x++)
    {
        for(int y = 0; y < dst.cols(); y++)
        {
            dst(0,x,y) = 1 - omega*dark_temp(0,x,y);
        }
    }
}

void TransmissionRefine(Image& im, Image& et, Image& dst)
{
    Image grey(im.rows(), im.cols(), 1);

    rgb2grey(im, grey);
    
    Guidedfilter(grey, et, 61, 0.0001, dst);
}


void Guidedfilter(Image& im, Image& p, int r, float eps, Image& dst)
{
    assert (r % 2 == 1);    // Only odd kernels supported

    Image mean_I(im.rows(), im.cols(), 1);
    Image mean_p(im.rows(), im.cols(), 1);
    boxFilter(im, mean_I, r, r);
    boxFilter(p, mean_p, r, r);

    Image Ip(im.rows(), im.cols(), 1);
    
    for(int x = 0; x < Ip.rows(); x++)
    {
        for(int y = 0; y < Ip.cols(); y++)
        {
            Ip(0,x,y) = im(0,x,y) * p(0,x,y);
        }
    }
    
    Image mean_Ip(im.rows(), im.cols(), 1);
    boxFilter(Ip, mean_Ip, r, r);
    
    Image cov_Ip(im.rows(), im.cols(), 1);
    
    for(int x = 0; x < cov_Ip.rows(); x++)
    {
        for(int y = 0; y < cov_Ip.cols(); y++)
        {
            cov_Ip(0,x,y) = mean_Ip(0,x,y) - (mean_I(0,x,y) * mean_p(0,x,y));
        }
    }
    
    Image imim(im.rows(), im.cols(), 1);
    for(int x = 0; x < imim.rows(); x++)
    {
        for(int y = 0; y < imim.cols(); y++)
        {
            imim(0,x,y) = im(0,x,y) * im(0,x,y);
        }
    }
    
    Image mean_II(im.rows(), im.cols(), 1);
    boxFilter(imim, mean_II, r, r);
    
    Image var_I(im.rows(), im.cols(), 1);
    for(int x = 0; x < var_I.rows(); x++)
    {
        for(int y = 0; y < var_I.cols(); y++)
        {
            var_I(0,x,y) = mean_II(0,x,y) - (mean_I(0,x,y) * mean_I(0,x,y));
        }
    }

    Image a(im.rows(), im.cols(), 1);
    for(int x = 0; x < a.rows(); x++)
    {
        for(int y = 0; y < a.cols(); y++)
        {
            a(0,x,y) = cov_Ip(0,x,y) / (var_I(0,x,y) + eps);
        }
    }
    
    Image b(im.rows(), im.cols(), 1);
    for(int x = 0; x < b.rows(); x++)
    {
        for(int y = 0; y < b.cols(); y++)
        {
            b(0,x,y) = mean_p(0,x,y) - (a(0,x,y) * mean_I(0,x,y));
        }
    }

    Image mean_a(im.rows(), im.cols(), 1);
    Image mean_b(im.rows(), im.cols(), 1);
    boxFilter(a, mean_a, r, r);
    boxFilter(b, mean_b, r, r);
    
    for(int x = 0; x < dst.rows(); x++)
    {
        for(int y = 0; y < dst.cols(); y++)
        {
            dst(0,x,y) = (mean_a(0,x,y)*im(0,x,y)) + mean_b(0,x,y);
        }
    }
}

void boxFilter(Image& src, Image& dst, int width, int height)
{
#ifndef CONV_KERNEL1D
    // create kernel
    float a = 1.0f/(width * height);
    Image kernel(width, width, 1);
    kernel.fill(a);
    
    // Convolute filter
    convoluteKernel(src, kernel, dst);
#else
    
    assert(width == height);
    
    // Can split a 2d kernel into 2 1D kernels, this average kernel
    // can be split to a 1 by N vector and one of N by 1 vector of
    // same value
    Image kernel1(1, width, 1);
    kernel1.fill(-1.0f/width);
    
    Image kernel2(width, 1, 1);
    kernel2.fill(-1.0f/width);
    
    // need an intermediate image
    Image src_dst(dst.rows(), dst.cols(), 1);
    
    // Convolute filter
    convoluteKernel(src,        kernel1, src_dst);
    convoluteKernel(src_dst,    kernel2, dst);

#endif
}

void convoluteKernel(Image& src, Image& kernel, Image& dst)
{
    int x_delta = floor(kernel.rows()/2);
    int y_delta = floor(kernel.cols()/2);
    
    // Run thru the dst image
    for(int x = 0; x < src.rows(); x++)
    {
        for(int y = 0; y < src.cols(); y++)
        {
            // Convolute the kernel
            float accumulator = 0.0f;
            for(int i = (x - x_delta), i_k = 0;
                i <= (x + x_delta);
                i++, i_k++)
            {
                for(int j = (y - y_delta), j_k = 0;
                    j <= (y + y_delta);
                    j++, j_k++)
                {
                    int i_fixed = i, j_fixed = j;
                    // Check bounds
                    if(i < 0)
                    {
                        // Out of average on the left side, abs(x) is enoug
                        i_fixed = abs(i);
                    }
                    if(j < 0)
                    {
                        // Out of average on the left side, abs(x) is enoug
                        j_fixed = abs(j);
                    }
                    if(i >= src.rows() )
                    {
                        // Out of average, substract distance from center
                        i_fixed = src.rows() - (i - src.rows() + 1);
                    }
                    if(j >= src.cols())
                    {
                        // Out of average, substract distance from center
                        j_fixed = src.cols() - (j - src.cols() + 1);
                    }
                    
                    accumulator += src(0,i_fixed,j_fixed) * kernel(0, i_k, j_k);
                }
            }
            
            // kernel has been run thru the image, lets write the MIN value
            dst(0,x,y) = accumulator;
        }
    }
}

void rgb2grey(Image& src, Image& dst)
{
    // Rec.ITU-R BT.601-7
    // 0.2989 * R + 0.5870 * G + 0.1140 * B
    for(int x = 0; x < dst.rows(); x++)
    {
        for(int y = 0; y < dst.cols(); y++)
        {
            dst(0,x,y) = 0.2989f*src(0,x,y) + 0.5870f*src(1,x,y) + 0.1140f*src(2,x,y);
        }
    }
}

void Recover(Image &im, Image &t, Image& dst, float* A, float tx=0.1)
{
    for(int x = 0; x < dst.rows(); x++)
    {
        for(int y = 0; y < dst.cols(); y++)
        {
            // not div/0
            t(0,x,y) = MAX(t(0,x,y), tx);
            
            dst(0,x,y) = (im(0,x,y) - A[0])/t(0,x,y) + A[0];
            dst(1,x,y) = (im(1,x,y) - A[1])/t(0,x,y) + A[1];
            dst(2,x,y) = (im(2,x,y) - A[2])/t(0,x,y) + A[2];
        }
    }
}

// Retunr a vector that contains the indices of the
// sorted elements, original vector is not modified
template<typename T>
std::vector<size_t> argsort(const std::vector<T> &array) {
    std::vector<size_t> indices(array.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&array](int left, int right) -> bool {
        // sort indices according to corresponding array element
        return array[left] < array[right];
    });
    
    return indices;
}

void imshow_float_array(char *name, Image& _im)
{
    cv::Mat I;
    
    switch (_im.chan())
    {
        case 3:
        {
            cv::Mat im_b(_im.rows(), _im.cols(), CV_32F, &_im(2,0,0));
            cv::Mat im_g(_im.rows(), _im.cols(), CV_32F, &_im(1,0,0));
            cv::Mat im_r(_im.rows(), _im.cols(), CV_32F, &_im(0,0,0));
            cv::Mat im_channels[3] = {im_b, im_g, im_r};
            
            cv::merge(im_channels, 3, I);
            break;
        }
        case 1:
        {
            cv::Mat temp(_im.rows(), _im.cols(), CV_32F, &_im(0,0,0));
            I = temp;
            break;
        }
        default:
        {
            cout << "Im contains channels: " << _im.chan() << " and its not suported" << endl;
        }
    }
    
    imshow(name, I);
}

int main(int argc, const char ** argv)
{
    
//    Image im(image_rows, image_cols, 3, &image[0][0][0]);
    Image im(560, 780, 3);
            
        auto start = high_resolution_clock::now();
        
        Image dark(im.rows(), im.cols(), 1);
        DarkChannel(im, dark, 15);
        
        auto _darkchannel = high_resolution_clock::now();
        
        float* airlight = (float *)malloc(3*sizeof(float));
        AtmLight(im, dark, airlight);
        
        auto _airlight = high_resolution_clock::now();
        
        
        Image te(im.rows(), im.cols(), 1);
        TransmissionEstimate(im, airlight, te,15);


        auto _transmision = high_resolution_clock::now();
        
        
        Image t(im.rows(), im.cols(), 1);
        TransmissionRefine(im,te,t);
        

        auto _transmision_refine = high_resolution_clock::now();

        Image J(im.rows(), im.cols(), 3);
        Recover(im, t, J, airlight, 0.1);
        
        auto stop = high_resolution_clock::now();
        
        auto duration_dc = duration_cast<milliseconds>(_darkchannel - start);
        auto duration_air = duration_cast<milliseconds>(_airlight - _darkchannel);
        auto duration_trams = duration_cast<milliseconds>(_transmision - _airlight);
        auto duration_tramsred = duration_cast<milliseconds>(_transmision_refine - _transmision);
        auto duration_stop = duration_cast<milliseconds>(stop - _transmision_refine);
        auto duration = duration_cast<milliseconds>(stop - start);
        
        cout << "Time taken by Dark channel:        "  << duration_dc.count() << " milliseconds" << endl;
        cout << "Time taken by airlight:            "  << duration_air.count() << " milliseconds" << endl;
        cout << "Time taken by transmision:         "  << duration_trams.count() << " milliseconds" << endl;
        cout << "Time taken by tranmsmision refined:"  << duration_tramsred.count() << " milliseconds" << endl;
        cout << "Time taken by recover :            "  << duration_stop.count() << " milliseconds" << endl;
        cout << "Time total:                        "  << duration.count() << " milliseconds" << endl;
        cout << endl << endl;
        
//        imshow_float_array("I", im);
    //    imshow_float_array("dark", dark);
    //    imshow_float_array("te", te);
    //    imshow_float_array("t", t);
//        imshow_float_array("J", J);
    //    waitKey(0);

    return 0;
}
