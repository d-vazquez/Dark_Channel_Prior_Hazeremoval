
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <stdio.h>
#include <iostream>

#include "raw_image.hpp"

using namespace cv;
using namespace std;


int main(int argc, const char ** argv)
{
    cv::Mat im_r(image_rows, image_cols, CV_32F, &image[0][0][0]);
    cv::Mat im_g(image_rows, image_cols, CV_32F, &image[1][0][0]);
    cv::Mat im_b(image_rows, image_cols, CV_32F, &image[2][0][0]);
    
    cv::Mat im_channels[3] = {im_b, im_g, im_r};
    cv::Mat I;
    
    cv::merge(im_channels, 3, I);
 
    imshow("Mat reader", I);
    waitKey(0);
    
    return 0;
}
