/* experimental custom ocl filter */


#ifndef __OPENCV_CUSTOM_OCL_FILTER_HPP__
#define __OPENCV_CUSTOM_OCL_FILTER_HPP__

#include "opencv2/core.hpp"

/*! \namespace cv
Namespace where all the C++ OpenCV functionality resides
*/
namespace cv {
namespace ocl {

class CV_EXPORTS CustomFilter : public Algorithm
{
public:
    static Ptr<CustomFilter> create(int ninputs, int noutputs, int borderType,
            String programSource = String(), Size localSize = Size());

    bool run(const std::vector<UMat>& inputs, std::vector<UMat>& outputs, Size imageSize);
    bool run(const UMat& input, UMat& output, Size imageSize);
};


} // namespace ocl
} // namespace cv

#endif // __OPENCV_CUSTOM_OCL_FILTER_HPP__
