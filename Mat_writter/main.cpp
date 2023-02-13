
#include "opencv2/core.hpp"
//#include "opencv2/imgproc.hpp"
//#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "mat_writer.hpp"

#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;


int main(int argc, const char ** argv)
{
    Mat I = imread("/Users/dariovazquez/fuente.png", IMREAD_COLOR);
    I.convertTo(I, CV_32FC3);
    cv::normalize(I, I, 0, 1, cv::NORM_MINMAX);
    
    write_mat_print(I);
    
    return 0;
}
