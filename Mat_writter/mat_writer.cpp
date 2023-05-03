//
//  mat_writer.c
//  Dark_Channel_Prior_Hazeremoval
//
//  Created by Dario Vazquez on 2/8/23.
//

#include <stdio.h>
#include <iostream>
#include "mat_writer.hpp"

using namespace cv;
using namespace std;

void write_mat_print(Mat &im)
{
    vector<Mat> Mat_ch(3);
    cv::split(im, Mat_ch);
    
    int rows = im.rows;
    int cols = im.cols;
    
    cout << endl;
    cout << "#ifndef raw_image_h" << endl;
    cout << "#define raw_image_h" << endl << endl;
    
    cout << "#define image_rows " << rows << endl;
    cout << "#define image_cols " << cols << endl << endl;
    
    cout << "float image[3]["<< rows << "]["<< cols << "] = ";
    cout << "{" << endl;
    cout << "// Red channel " << endl;
    cout << " { " << endl;
    for(int i = 0; i < rows; i++)
    {
        cout << "\t{" ;
        for(int j = 0; j < cols; j++)
        {
            cout << Mat_ch[2].at<float>(i,j) << ", ";
        }
        cout << "}, " << endl;
    }
    cout << "}," << endl << endl;
    
    
//    cout << "float image_g[" << rows << "][" << cols << "] = ";
    cout << "// Green channel " << endl;
    cout << "{ " << endl;
    for(int i = 0; i < rows; i++)
    {
        cout << "\t{" ;
        for(int j = 0; j < cols; j++)
        {
            cout << Mat_ch[1].at<float>(i,j) << ", ";
        }
        cout << "}, " << endl;
    }
    cout << "}," << endl << endl;
    
    
//    cout << "float image_b[" << rows << "][" << cols << "] = ";
    cout << "// Blue channel " << endl;
    cout << "{ " << endl;
    for(int i = 0; i < rows; i++)
    {
        cout << "\t{" ;
        for(int j = 0; j < cols; j++)
        {
            cout << Mat_ch[0].at<float>(i,j) << ", ";
        }
        cout << "}, " << endl;
    }
    cout << "}," << endl << endl;
    cout << "};" << endl;
    
    
    cout << "#endif" << endl;
}


void write_mat_print_int(Mat &im)
{
    vector<Mat> Mat_ch(3);
    cv::split(im, Mat_ch);
    
    int rows = im.rows;
    int cols = im.cols;
    
    cout << endl;
    cout << "#ifndef raw_image_h" << endl;
    cout << "#define raw_image_h" << endl << endl;
    
    cout << "#include <stdint.h>" << endl;
    
    cout << "#define image_rows " << rows << endl;
    cout << "#define image_cols " << cols << endl << endl;
    
    cout << "const uint8_t image[3]["<< rows << "]["<< cols << "] = ";
    cout << "{" << endl;
    cout << "// Red channel " << endl;
    cout << " { " << endl;
    for(int i = 0; i < rows; i++)
    {
        cout << "\t{" ;
        for(int j = 0; j < cols; j++)
        {
            cout << (int)Mat_ch[2].at<uchar>(i,j) << ", ";
        }
        cout << "}, " << endl;
    }
    cout << "}," << endl << endl;
    
    
//    cout << "float image_g[" << rows << "][" << cols << "] = ";
    cout << "// Green channel " << endl;
    cout << "{ " << endl;
    for(int i = 0; i < rows; i++)
    {
        cout << "\t{" ;
        for(int j = 0; j < cols; j++)
        {
            cout << (int)Mat_ch[1].at<uchar>(i,j) << ", ";
        }
        cout << "}, " << endl;
    }
    cout << "}," << endl << endl;
    
    
//    cout << "float image_b[" << rows << "][" << cols << "] = ";
    cout << "// Blue channel " << endl;
    cout << "{ " << endl;
    for(int i = 0; i < rows; i++)
    {
        cout << "\t{" ;
        for(int j = 0; j < cols; j++)
        {
            cout << (int)Mat_ch[0].at<uchar>(i,j) << ", ";
        }
        cout << "}, " << endl;
    }
    cout << "}," << endl << endl;
    cout << "};" << endl;
    
    
    cout << "#endif" << endl;
}
