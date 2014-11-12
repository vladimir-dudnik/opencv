/* experimental custom ocl filter */

#include "precomp.hpp"
#include "opencv2/custom_ocl_filter.hpp"
#include "opencl_kernels_imgproc.hpp"


namespace cv {
namespace ocl {

class CustomFilterImpl : public CustomFilter
{
public:
    CustomFilterImpl(
        int    ninputs,
        int    noutputs,
        int    borderType,
        String programSource,
        String buildOptions,
        Size   localSize)
    {
        Size sz = localSize;
        size_t globalsize[2] = { sz.width, sz.height };
        size_t localsize_general[2] = { 0, 1 };

        const Device& device = Device::getDefault();

        size_t tryWorkItems = device.maxWorkGroupSize();
        if (device.isIntel() && 128 < tryWorkItems)
            tryWorkItems = 128;
/*
        char build_options[1024];
        sprintf(build_options, "-D cn=%d "
        "-D ANCHOR_X=%d -D ANCHOR_Y=%d -D KERNEL_SIZE_X=%d -D KERNEL_SIZE_Y=%d "
        "-D PX_LOAD_VEC_SIZE=%d -D PX_LOAD_NUM_PX=%d "
        "-D PX_PER_WI_X=%d -D PX_PER_WI_Y=%d -D PRIV_DATA_WIDTH=%d -D %s -D %s "
        "-D PX_LOAD_X_ITERATIONS=%d -D PX_LOAD_Y_ITERATIONS=%d "
        "-D srcT=%s -D srcT1=%s -D dstT=%s -D dstT1=%s -D WT=%s -D WT1=%s "
        "-D convertToWT=%s -D convertToDstT=%s %s",
        cn,
        anchor.x, anchor.y, ksize.width, ksize.height,
        pxLoadVecSize, pxLoadNumPixels,
        pxPerWorkItemX, pxPerWorkItemY, privDataWidth, borderMap[borderType],
        isolated ? "BORDER_ISOLATED" : "NO_BORDER_ISOLATED",
        privDataWidth / pxLoadNumPixels, pxPerWorkItemY + ksize.height - 1,
        ocl::typeToStr(type), ocl::typeToStr(sdepth), ocl::typeToStr(dtype),
        ocl::typeToStr(ddepth), ocl::typeToStr(wtype), ocl::typeToStr(wdepth),
        ocl::convertTypeStr(sdepth, wdepth, cn, cvt[0]),
        ocl::convertTypeStr(wdepth, ddepth, cn, cvt[1]), kerStr.c_str());
*/
        ProgramSource prg(programSource);
        if (!k.create("my_kernel", prg, buildOptions))
            return;

        return;
    }

    virtual bool run(const UMat& input, UMat& output, Size imageSize)
    {
        globalSize[0] = imageSize.width;// / predictOptimalVectorWidth(input, output);
        globalSize[1] = (imageSize.height + 4 - 1) / 4;
        localSize = 2;
        int srcStep = input.step[0];
        int dstStep = output.step[0];
        k.args(KernelArg::PtrReadOnly(input), srcStep, KernelArg::PtrWriteOnly(output), dstStep, imageSize.width, imageSize.height);
        k.run(2, globalSize, 0, false);
        return true;
    }

private:
    size_t globalSize[2];
    size_t localSize;
    Kernel k;
};


const char * const borderMap[] =
{
    "BORDER_CONSTANT",
    "BORDER_REPLICATE",
    "BORDER_REFLECT",
    "BORDER_WRAP",
    "BORDER_REFLECT_101"
};

Ptr<CustomFilter> CustomFilter::create(
    int    ninputs,
    int    noutputs,
    int    borderType,
    String programSource,
    String buildOptions,
    Size   localSize)
{
    return makePtr<CustomFilterImpl>(ninputs, noutputs, borderType, programSource, buildOptions, localSize);
}


//bool CustomFilter::run(const std::vector<UMat>& inputs, std::vector<UMat>& outputs, Size imageSize)
//{
//    return true;
//}


//bool CustomFilter::run(const UMat& input, UMat& output, Size imageSize)
//{
//    return true;
//}

} // namespace ocl
} // namespace cv

