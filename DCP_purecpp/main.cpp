
#include "opencv2/core.hpp"
//#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
//#include "opencv2/ximgproc/edge_filter.hpp"

#include <stdio.h>
#include <iostream>
#include <math.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <thread>
#include <functional>


#include "raw_image.hpp"

#include "Image.hpp"


#ifndef MIN
#  define MIN(a,b)  ((a) > (b) ? (b) : (a))
#endif

#ifndef MAX
#  define MAX(a,b)  ((a) < (b) ? (b) : (a))
#endif

#define CAP_VALUE(A) MAX(MIN(A,1.0f),0.0f);

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


const Mat DarkChannel(Mat &img, int sz);

const Scalar AtmLight(Mat &img, Mat &dark);
const Mat  TransmissionEstimate(Mat &im, Scalar A, int sz);
const Mat TransmissionRefine(Mat &im, Mat &et);
const Mat Guidedfilter(Mat &im, Mat &p, int r, int eps);
const Mat Recover(Mat &im, Mat &t, Scalar A, float tx);

static
void parallel_for(unsigned nb_elements,
                  std::function<void (int start, int end)> functor,
                  bool use_threads = true)
{
    // -------
    unsigned nb_threads_hint = std::thread::hardware_concurrency();
    unsigned nb_threads = nb_threads_hint == 0 ? 8 : (nb_threads_hint);

    unsigned batch_size = nb_elements / nb_threads;
    unsigned batch_remainder = nb_elements % nb_threads;

    std::vector< std::thread > my_threads(nb_threads);

    if( use_threads )
    {
        // Multithread execution
        for(unsigned i = 0; i < nb_threads; ++i)
        {
            int start = i * batch_size;
            my_threads[i] = std::thread(functor, start, start+batch_size);
        }
    }
    else
    {
        // Single thread execution (for easy debugging)
        for(unsigned i = 0; i < nb_threads; ++i){
            int start = i * batch_size;
            functor( start, start+batch_size );
        }
    }

    // Deform the elements left
    int start = nb_threads * batch_size;
    functor( start, start+batch_remainder);

    // Wait for the other thread to finish their task
    if( use_threads )
        std::for_each(my_threads.begin(), my_threads.end(), std::mem_fn(&std::thread::join));
}

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
    
    // Create erode kernel
    Image kernel(sz, sz, 1);
    
#pragma omp parallel for
    for(int i = 0; i < kernel.rows(); i++)
    {
        for(int j = 0; j < kernel.cols(); j++)
        {
            kernel(0,i,j) = 1.0f;
        }
    }
    
    erode(dc, kernel, dark);
}

void erode(Image& src, Image& kernel, Image& dst)
{
    int x_delta = floor(kernel.rows()/2);
    int y_delta = floor(kernel.cols()/2);
    
    // Run thru the dst image
    for(int x = 0; x < dst.rows(); x++)
    {
        for(int y = 0; y < dst.cols(); y++)
        {
            // Convolute the kernel
            
            // we need to keep track of the minimum in the kernel, the max value of
            // the image can be 1.0, so lets start there
            float erode_value = 1.0f;
            for(int i = (x - x_delta); i < (x + x_delta); i++)
            {
                for(int j = (y - y_delta); j < (y + y_delta); j++)
                {
                    // Check bounds
                    if(i < 0 || j < 0 || i >= src.rows() || j >= src.cols())
                    {
                        continue;
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
    
    //    Mat gray;
    //    cvtColor(im, gray, cv::COLOR_BGR2GRAY);
    
    rgb2grey(im, grey);
    
    
    //    return Guidedfilter(gray, et, 60, 0.0001);
    
    Guidedfilter(grey, et, 60, 0.0001, dst);
    
    
//    imshow_float_array("im", im);
//    imshow_float_array("grey", grey);
//    imshow_float_array("dst", dst);
//    waitKey(0);
//    cv::destroyAllWindows();
    
    
}


void Guidedfilter(Image& im, Image& p, int r, float eps, Image& dst)
{
    //    Mat q, mean_I, mean_p, mean_Ip;
    //    Mat mean_II, mean_a, mean_b;
    //    Mat im_p;
    //
    //    cv::boxFilter(im, mean_I, CV_32F, Size(r,r));
    //    cv::boxFilter(p, mean_p, CV_32F, Size(r,r));
    Image mean_I(im.rows(), im.cols(), 1);
    Image mean_p(im.rows(), im.cols(), 1);
    boxFilter(im, mean_I, r, r);
    boxFilter(p, mean_p, r, r);
    
    //    imshow_float_array("im", im);
    
    
    
    
    //
    //    cv::boxFilter(im.mul(p), mean_Ip, CV_32F, Size(r,r));
    Image Ip(im.rows(), im.cols(), 1);
    //= im*p;
    
    for(int x = 0; x < Ip.rows(); x++)
    {
        for(int y = 0; y < Ip.cols(); y++)
        {
            Ip(0,x,y) = im(0,x,y) * p(0,x,y);
        }
    }
    
    Image mean_Ip(im.rows(), im.cols(), 1);
    boxFilter(Ip, mean_Ip, r, r);
    
    
    //    Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);
    Image cov_Ip(im.rows(), im.cols(), 1);
    
    for(int x = 0; x < cov_Ip.rows(); x++)
    {
        for(int y = 0; y < cov_Ip.cols(); y++)
        {
            cov_Ip(0,x,y) = mean_Ip(0,x,y) - (mean_I(0,x,y) * mean_p(0,x,y));
        }
    }
    
    //    cv::boxFilter(im.mul(im), mean_II,CV_32F,Size(r,r));
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
    
    
    //    Mat var_I = mean_II - mean_I.mul(mean_I);
    
    Image var_I(im.rows(), im.cols(), 1);
    for(int x = 0; x < var_I.rows(); x++)
    {
        for(int y = 0; y < var_I.cols(); y++)
        {
            var_I(0,x,y) = mean_II(0,x,y) - (mean_I(0,x,y) * mean_I(0,x,y));
        }
    }
    
    
    
    
    //    Mat a = cov_Ip/(var_I + eps);
    //    Mat b = mean_p - a.mul(mean_I);
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
    
    
    
//
    
//    cv::boxFilter(a, mean_a, CV_32F, Size(r,r));
//    cv::boxFilter(b, mean_b, CV_32F, Size(r,r));
    Image mean_a(im.rows(), im.cols(), 1);
    Image mean_b(im.rows(), im.cols(), 1);
    boxFilter(a, mean_a, r, r);
    boxFilter(b, mean_b, r, r);
    
//
//    imshow_float_array("a", a);
//    imshow_float_array("b", b);
//    imshow_float_array("mean_a", mean_a);
//    imshow_float_array("mean_b", mean_b);
//    waitKey(0);
    
//    q = im.mul(mean_a) + mean_b;
//    return q;
    for(int x = 0; x < dst.rows(); x++)
    {
        for(int y = 0; y < dst.cols(); y++)
        {
            dst(0,x,y) = (mean_a(0,x,y)*im(0,x,y)) + mean_b(0,x,y);
        }
    }
    
//    imshow_float_array("dst", dst);
//    waitKey(0);
}

void boxFilter(Image& src, Image& dst, int width, int height)
{
    // create kernel
    float a = 1.0f/(width * height);
    Image kernel(width, width, 1);
    kernel.fill(a);
    
    // Convolute filter
    convoluteKernel(src, kernel, dst);
}

void convoluteKernel(Image& src, Image& kernel, Image& dst)
{
    int x_delta = floor(kernel.rows()/2);
    int y_delta = floor(kernel.cols()/2);
    
//    if(!kernel.rows()%2)
//    {
//
//    }
//
    // Run thru the dst image
    for(int x = 0; x < src.rows(); x++)
    {
        for(int y = 0; y < src.cols(); y++)
        {
            // Convolute the kernel
            float accumulator = 0.0f;
            for(int i = (x - x_delta), i_k = 0;
                i < (x + x_delta);
                i++, i_k++)
            {
                for(int j = (y - y_delta), j_k = 0;
                    j < (y + y_delta);
                    j++, j_k++)
                {
                    int i_fixed = i, j_fixed = j;
                    // Check bounds
                    if(i < 0)
                    {
                        // Out of average on the left side, abs(x) is enoug
                        i_fixed = abs(i);
                    }
                    if(i < 0 || j < 0 )
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
//    Mat res;
//
//    t = cv::max(t, tx); // Make sure t does not contain 0
//
//    // perform operation on each channel
//    vector<Mat> res_ch(3);
//    vector<Mat> im_ch(3);
//    cv::split(im, im_ch);
//
//    res_ch[0] = (im_ch[0] - A.val[0])/t + A.val[0];
//    res_ch[1] = (im_ch[1] - A.val[1])/t + A.val[1];
//    res_ch[2] = (im_ch[2] - A.val[2])/t + A.val[2];
//
//    cv::merge(res_ch, res);
//
//    return res;
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



//--------------------------------------------------
const Mat DarkChannel(Mat &img, int sz)
{
    // Split image into b, g, r channels
    vector<Mat> channels(3);
    cv::split(img, channels);
    
    // Obtaining dark channel, so each pixel's min value
    Mat dc = cv::min(cv::min(channels[0], channels[1]), channels[2]);
    
    // 'erode' image, so calculate the minimun value in the window given by sz
    Mat kernel = getStructuringElement(cv::MorphShapes::MORPH_RECT, Size(sz,sz));
    Mat dark = Mat::zeros(img.rows, img.cols, CV_32FC3);
    cv::erode(dc, dark, kernel);
    
    return dark;
}


const Scalar AtmLight(Mat &im, Mat &dark)
{
    int _rows = im.rows;
    int _cols = im.cols;
    
    int imsz = _rows*_cols;
    
    int numpx = (int)MAX(imsz/1000, 1);
    
    Mat darkvec = dark.reshape(0, 1);
    Mat imvec = im.reshape(0, 1);
    
    Mat indices;
    cv::sortIdx(darkvec, indices, SORT_DESCENDING);
    
    Scalar atmsum(0, 0, 0, 0);
    for(int ind = 0; ind < numpx; ind++)
    {
        atmsum.val[0] += imvec.at<Vec3f>(0, indices.at<int>(0,ind))[0];
        atmsum.val[1] += imvec.at<Vec3f>(0, indices.at<int>(0,ind))[1];
        atmsum.val[2] += imvec.at<Vec3f>(0, indices.at<int>(0,ind))[2];
    }
    
    atmsum.val[0] = atmsum.val[0]/ numpx;
    atmsum.val[1] = atmsum.val[1]/ numpx;
    atmsum.val[2] = atmsum.val[2]/ numpx;
    
    return atmsum;
}

const Mat  TransmissionEstimate(Mat &im, Scalar A, int sz)
{
    float omega = 0.95;
    
    Mat im3;
    
    
    vector<Mat> img3_ch(3);
    vector<Mat> im_ch(3);
    cv::split(im, img3_ch);
    cv::split(im, im_ch);
    
    img3_ch[0] = im_ch[0] / A.val[0];
    img3_ch[1] = im_ch[1] / A.val[1];
    img3_ch[2] = im_ch[2] / A.val[2];
    
    cv::merge(img3_ch, im3);
    
    Mat transmission = 1 - omega*DarkChannel(im3,sz);
    
    return transmission;
}


const Mat TransmissionRefine(Mat &im, Mat &et)
{
    Mat gray;
    cvtColor(im, gray, cv::COLOR_BGR2GRAY);
    return Guidedfilter(gray, et, 60, 0.0001);
}

const Mat Guidedfilter(Mat &im, Mat &p, int r, int eps)
{
    Mat q, mean_I, mean_p, mean_Ip;
    Mat mean_II, mean_a, mean_b;
    Mat im_p;
    
    cv::boxFilter(im, mean_I, CV_32F, Size(r,r));
    cv::boxFilter(p, mean_p, CV_32F, Size(r,r));
    
    cv::boxFilter(im.mul(p), mean_Ip, CV_32F, Size(r,r));
    
    Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);
    
    cv::boxFilter(im.mul(im), mean_II,CV_32F,Size(r,r));
    Mat var_I = mean_II - mean_I.mul(mean_I);
    
    Mat a = cov_Ip/(var_I + eps);
    Mat b = mean_p - a.mul(mean_I);
    
    cv::boxFilter(a, mean_a, CV_32F, Size(r,r));
    cv::boxFilter(b, mean_b, CV_32F, Size(r,r));
    
    q = im.mul(mean_a) + mean_b;
    return q;
}


const Mat Recover(Mat &im, Mat &t, Scalar A, float tx=0.1)
{
    Mat res;
    
    t = cv::max(t, tx); // Make sure t does not contain 0
    
    // perform operation on each channel
    vector<Mat> res_ch(3);
    vector<Mat> im_ch(3);
    cv::split(im, im_ch);
    
    res_ch[0] = (im_ch[0] - A.val[0])/t + A.val[0];
    res_ch[1] = (im_ch[1] - A.val[1])/t + A.val[1];
    res_ch[2] = (im_ch[2] - A.val[2])/t + A.val[2];
    
    cv::merge(res_ch, res);
    
    return res;
}


// _im[2*image_rows*image_cols + 1*image_cols + 1]
// _im,image_rows,image_cols,    2,1,1)


int main(int argc, const char ** argv)
{
    Image im(image_rows, image_cols, 3, &image[0][0][0]);
    
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
    
//    imshow_float_array("I", im);
//    imshow_float_array("dark", dark);
//    imshow_float_array("te", te);
//    imshow_float_array("t", t);
//    imshow_float_array("J", J);
//    waitKey(0);
  
    
    return 0;
}
