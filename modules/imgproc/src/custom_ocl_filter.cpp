/* experimental custom ocl filter */

#include "precomp.hpp"
#include "opencv2/custom_ocl_filter.hpp"


namespace cv {
namespace ocl {

Ptr<CustomFilter> CustomFilter::create(int ninputs, int noutputs, int borderType,
    String programSource, Size localSize)
{
    CustomFilter* filter = new CustomFilter;
    return Ptr<CustomFilter>(filter);
}


bool CustomFilter::run(const std::vector<UMat>& inputs, std::vector<UMat>& outputs, Size imageSize)
{
    return true;
}


bool CustomFilter::run(const UMat& input, UMat& output, Size imageSize)
{
    return true;
}

} // namespace ocl
} // namespace cv

