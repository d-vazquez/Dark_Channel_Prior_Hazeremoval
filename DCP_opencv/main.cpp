
#include "opencv2/core.hpp"
//#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
//#include "opencv2/ximgproc/edge_filter.hpp"

#include <stdio.h>
#include <iostream>
#include <math.h>

using namespace std::chrono;
using namespace cv;
using namespace std;

const Mat KNB(Mat &mw, int K);

static void help(const char ** argv)
{
    printf("Usage:\n %s [image_name]\n",argv[0]);
}

const char* keys =
{
    "{help h usage ? |      |   }"
    "{@image1        |   /Users/dariovazquez/fuente.png  | image1 for process   }"
};

const Mat DarkChannel(Mat &img, int sz);
const Scalar AtmLight(Mat &img, Mat &dark);
const Mat  TransmissionEstimate(Mat &im, Scalar A, int sz);
const Mat TransmissionRefine(Mat &im, Mat &et);
const Mat Guidedfilter(Mat &im, Mat &p, int r, int eps);
const Mat Recover(Mat &im, Mat &t, Scalar A, float tx);

const Mat DarkChannel(Mat &img, int sz)
{
    // Split image into b, g, r channels
    vector<Mat> channels(3);
    cv::split(img, channels);
    
    // Obtaining dark channel, so each pixel's min value
    Mat dc = cv::min(cv::min(channels[0], channels[1]), channels[2]);
    
    // 'erode' image, so calculate the minimun value in the window given by sz
    Mat kernel = getStructuringElement(cv::MorphShapes::MORPH_RECT, Size(sz,sz));
    
    Mat dark;
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
        atmsum.val[0] += imvec.at<Vec3b>(0, indices.at<int>(0,ind))[0];
        atmsum.val[1] += imvec.at<Vec3b>(0, indices.at<int>(0,ind))[1];
        atmsum.val[2] += imvec.at<Vec3b>(0, indices.at<int>(0,ind))[2];
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
    
    img3_ch[0] = (im_ch[0] / A.val[0]) * 255;
    img3_ch[1] = (im_ch[1] / A.val[1]) * 255;
    img3_ch[2] = (im_ch[2] / A.val[2]) * 255;
    
    cv::merge(img3_ch, im3);
    
    Mat transmission = 255 - omega*DarkChannel(im3,sz);
    
    
    return transmission;
}


const Mat TransmissionRefine(Mat &im, Mat &et)
{
    Mat gray;
    cvtColor(im, gray, cv::COLOR_BGR2GRAY);
    return Guidedfilter(gray, et, 60, 1);
}

const Mat Guidedfilter(Mat &im, Mat &p, int r, int eps)
{
    cout << im.type() << endl;
    
    cout << "im before = " << endl << im(Range(0,5),Range(0,5)) << endl;
    
    im.convertTo(im, CV_32FC3);
    cv::normalize(im, im, 0, 1, cv::NORM_MINMAX);
    
    p.convertTo(p, CV_32FC1);
    cv::normalize(p, p, 0, 1, cv::NORM_MINMAX);
    
    
    cout << "im after = " << endl << im(Range(0,5),Range(0,5)) << endl;
    
    cout << im.type() << endl;
    
    Mat q, mean_I, mean_p, mean_Ip;
    Mat mean_II, mean_a, mean_b;
    Mat im_p;
    
    cv::boxFilter(im, mean_I, CV_32F, Size(r,r));
    
//    imshow("im", im);
//    imshow("mean_I", mean_I);
//    waitKey(0);
    
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
    
    
    
    cout << "q before = " << endl << q(Range(0,5),Range(0,5)) << endl;
    
    cv::normalize(q, q, 0, 255, cv::NORM_MINMAX);

    cout << "q normalize = " << endl << q(Range(0,5),Range(0,5)) << endl;

    
    q.convertTo(q, CV_8UC1);
    
    
    cout << "q after = " << endl << q(Range(0,5),Range(0,5)) << endl;
    
    cout << im.type() << endl;
    
    return q;
}


const Mat Recover(Mat &im, Mat &t, Scalar A, float tx=0.1)
{
    cout << im.type() << endl;
    cout << t.type() << endl;
//
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
    
    cout << "im = " << endl << im(Range(0,5),Range(0,5)) << endl;
    cout << "t = " << endl << t(Range(0,5),Range(0,5)) << endl;
    cout << "A = " << endl << A.val[0] << endl;
    
    Mat res;
    res.create(im.rows, im.cols, im.type());
    
    for(int _row = 0; _row < res.rows; _row++)
    {
        for(int _col = 0; _col < res.cols; _col++)
        {
            float factor = 255.f/t.at<uchar>(_row, _col);
            
            res.at<Vec3b>(_row, _col)[0] = (im.at<Vec3b>(_row, _col)[0] - A.val[0])*factor + A.val[0];
            res.at<Vec3b>(_row, _col)[1] = (im.at<Vec3b>(_row, _col)[1] - A.val[1])*factor + A.val[1];
            res.at<Vec3b>(_row, _col)[2] = (im.at<Vec3b>(_row, _col)[2] - A.val[2])*factor + A.val[2];
            
        }
    }
    
    
    return res;
}

#include "raw_image.hpp"

int main(int argc, const char ** argv)
{
    CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        help(argv);
        return 0;
    }

    if (!parser.has("@image1"))
    {
        help(argv);
        return 0;
    }
    
    // Read image
    string filename = parser.get<string>(0);
    
    Mat I = imread(filename, IMREAD_COLOR);
//    I.convertTo(I, CV_32FC3);
//    cv::normalize(I, I, 0, 1, cv::NORM_MINMAX);
    
    auto start = high_resolution_clock::now();
    
    Mat dark    = DarkChannel(I,15);
//    cout << "dark = " << endl << dark(Range(0,5),Range(0,5)) << endl;
//    imshow("dark", dark);
//    waitKey(0);
    
    auto _darkchannel = high_resolution_clock::now();
    
    Scalar A    = AtmLight(I,dark);
    
    auto _airlight = high_resolution_clock::now();
    
    Mat te      = TransmissionEstimate(I,A,15);
//    cout << "te = " << endl << te(Range(0,5),Range(0,5)) << endl;
//    imshow("te", te);
//    waitKey(0);
    
    auto _transmision = high_resolution_clock::now();
    
    Mat t       = TransmissionRefine(I,te);
//    cout << "t = " << endl << t(Range(0,5),Range(0,5)) << endl;
//    imshow("t", t);
//    waitKey(0);
    
    auto _transmision_refine = high_resolution_clock::now();
    
    Mat J       = Recover(I, t, A, 0.1);
    
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
    
    imshow("I", I);
    imshow("dark", dark);
    imshow("te", te);
    imshow("t", t);
    imshow("J", J);
    waitKey(0);
    
    return 0;
}
