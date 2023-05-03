
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"

#include <stdio.h>
#include <iostream>
#include <math.h>
#include <vector>
#include <numeric>

#include "raw_image.hpp"
#include "Image.hpp"
#include "dehaze.hpp"


using namespace std::chrono;
using namespace cv;
using namespace std;




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
    cout << endl << endl;
    
    imshow_float_array("I", im);
    imshow_float_array("dark", dark);
    imshow_float_array("te", te);
    imshow_float_array("t", t);
    imshow_float_array("J", J);
    waitKey(0);
    
    return 0;
}
