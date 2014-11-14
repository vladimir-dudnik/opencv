#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/custom_ocl_filter.hpp"

using namespace std;
using namespace cv;

class App
{
public:
    App(CommandLineParser& cmd);
    void run();
    void handleKey(char key);
    void filterWorkBegin();
    void filterWorkEnd();
    string filterWorkFps() const;
    void workBegin();
    void workEnd();
    string workFps() const;
    string message() const;


    // This function test if gpu_rst matches cpu_rst.
    // If the two vectors are not equal, it will return the difference in vector size
    // Else if will return
    // (total diff of each cpu and gpu rects covered pixels)/(total cpu rects covered pixels)
    double checkRectSimilarity(Size sz,
        std::vector<Rect>& cpu_rst,
        std::vector<Rect>& gpu_rst);
private:
    App operator=(App&);

    //Args args;
    bool running;
    bool make_gray;
    double scale;
    double resize_scale;
    int win_width;
    int win_stride_width;
    int win_stride_height;
    int gr_threshold;
    int nlevels;
    double hit_threshold;
    bool gamma_corr;

    int64 filter_work_begin;
    double filter_work_fps;
    int64 work_begin;
    double work_fps;

    string img_source;
    string vdo_source;
    string output;
    int camera_id;
    bool write_once;
};

int main(int argc, char** argv)
{
    const char* keys =
        "{ h help      | false          | print help message }"
        "{ i input     |                | specify input image}"
        "{ c camera    | -1             | enable camera capturing }"
        "{ v video     | 768x576.avi    | use video as input }"
        "{ g gray      | false          | convert image to gray one or not}"
        "{ s scale     | 1.0            | resize the image before detect}"
        "{ o output    |                | specify output path when input is images}";
    CommandLineParser cmd(argc, argv, keys);
    if (cmd.has("help"))
    {
        cout << "Usage : hog [options]" << endl;
        cout << "Available options:" << endl;
        cmd.printMessage();
        return EXIT_SUCCESS;
    }

    App app(cmd);
    try
    {
        app.run();
    }
    catch (const Exception& e)
    {
        return cout << "error: " << e.what() << endl, 1;
    }
    catch (const exception& e)
    {
        return cout << "error: " << e.what() << endl, 1;
    }
    catch (...)
    {
        return cout << "unknown exception" << endl, 1;
    }
    return EXIT_SUCCESS;
}

App::App(CommandLineParser& cmd)
{
    cout << "\nControls:\n"
        << "\tESC - exit\n"
        << "\tm - change mode GPU <-> CPU\n"
        << "\tg - convert image to gray or not\n"
        << "\to - save output image once, or switch on/off video save\n"
        << "\t1/q - increase/decrease HOG scale\n"
        << "\t2/w - increase/decrease levels count\n"
        << "\t3/e - increase/decrease HOG group threshold\n"
        << "\t4/r - increase/decrease hit threshold\n"
        << endl;

    make_gray = cmd.has("gray");
    resize_scale = cmd.get<double>("s");
    vdo_source = cmd.get<string>("v");
    img_source = cmd.get<string>("i");
    output = cmd.get<string>("o");
    camera_id = cmd.get<int>("c");

    win_width = 48;
    win_stride_width = 8;
    win_stride_height = 8;
    gr_threshold = 8;
    nlevels = 13;
    hit_threshold = 1.4;
    scale = 1.05;
    gamma_corr = true;
    write_once = false;

    cout << "Group threshold: " << gr_threshold << endl;
    cout << "Levels number: " << nlevels << endl;
    cout << "Win width: " << win_width << endl;
    cout << "Win stride: (" << win_stride_width << ", " << win_stride_height << ")\n";
    cout << "Hit threshold: " << hit_threshold << endl;
    cout << "Gamma correction: " << gamma_corr << endl;
    cout << endl;
}

void App::run()
{
    running = true;
    VideoWriter video_writer;

    Size win_size(win_width, win_width * 2);
    Size win_stride(win_stride_width, win_stride_height);

    // Create custom filter

    String kernel_source =
        "__kernel void my_kernel("
        "  __global const unsigned char* pSrc, int srcStep,"
        "  __global       unsigned char* pDst, int dstStep,"
        "  int width, int height)"
        "{"
        "  int x = get_global_id(0);"
        "  int y = get_global_id(1)*4;"
        "  if(x < width)"
        "  {"
        "    int srcIdx = mad24(y,srcStep,x);"
        "    int dstIdx = mad24(y,dstStep,x);"
        "    for(int i = y; i < y + 4; i++)"
        "    {"
        "      pDst[dstIdx] = pSrc[srcIdx]+1;"
        "      srcIdx += srcStep;"
        "      dstIdx += dstStep;"
        "    }"
        "  }"
        "}";
    String build_options = "";
    Ptr<ocl::CustomFilter> pFilter = ocl::CustomFilter::create(1, 1, BORDER_CONSTANT, kernel_source, build_options, Size(2, 2));

    while (running)
    {
        VideoCapture vc;
        UMat frame;

        if (vdo_source != "")
        {
            vc.open(vdo_source.c_str());
            if (!vc.isOpened())
                throw runtime_error(string("can't open video file: " + vdo_source));
            vc >> frame;
        }
        else if (camera_id != -1)
        {
            vc.open(camera_id);
            if (!vc.isOpened())
            {
                stringstream msg;
                msg << "can't open camera: " << camera_id;
                throw runtime_error(msg.str());
            }
            vc >> frame;
        }
        else
        {
            imread(img_source).copyTo(frame);
            if (frame.empty())
                throw runtime_error(string("can't open image file: " + img_source));
        }

        UMat img_aux;
        UMat img;
        Mat img_to_show;

        // Iterate over all frames
        while (running && !frame.empty())
        {
            workBegin();

            // Change format of the image
            if (make_gray)
                cvtColor(frame, img_aux, COLOR_BGR2GRAY);
            else
                frame.copyTo(img_aux);

            // Resize image
            if (abs(scale - 1.0) > 0.001)
            {
                Size sz((int)((double)img_aux.cols / resize_scale), (int)((double)img_aux.rows / resize_scale));
                resize(img_aux, img, sz);
            }
            else
                img = img_aux;

            // Perform HOG classification
            filterWorkBegin();

            UMat out;
            UMat kernel(3, 3, CV_32FC1);
            kernel.setTo(1);

            filter2D(img, out, -1, kernel, Point(-1, -1));

            Size sz(img.cols, img.rows);
            UMat dst;
            dst.create(sz, img.type());
            dst.setTo(0xff);

//            pFilter->run(img, dst, sz);

            filterWorkEnd();

            // Draw result
            Mat source;
            Mat aaa;
            img.copyTo(source);
            out.copyTo(img_to_show);
//            dst.copyTo(aaa);
            putText(img_to_show, ocl::useOpenCL() ? "Mode: OpenCL" : "Mode: CPU", Point(5, 25), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
            putText(img_to_show, "FPS (filter only): " + filterWorkFps(), Point(5, 65), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
            putText(img_to_show, "FPS (total): " + workFps(), Point(5, 105), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
            imshow("source", source);
            imshow("result", img_to_show);
//            imshow("aaa", aaa);
            if (vdo_source != "" || camera_id != -1)
                vc >> frame;

            workEnd();

            if (output != "" && write_once)
            {
                if (img_source != "")     // write image
                {
                    write_once = false;
                    imwrite(output, img_to_show);
                }
                else                      // write video
                {
                    if (!video_writer.isOpened())
                    {
                        video_writer.open(output, VideoWriter::fourcc('x', 'v', 'i', 'd'), 24,
                            img_to_show.size(), true);
                        if (!video_writer.isOpened())
                            throw std::runtime_error("can't create video writer");
                    }

                    if (make_gray)
                        cvtColor(img_to_show, img, COLOR_GRAY2BGR);
                    else
                        cvtColor(img_to_show, img, COLOR_BGRA2BGR);

                    video_writer << img.getMat(ACCESS_READ);
                }
            }

            handleKey((char)waitKey(3));
        }
    }
}

void App::handleKey(char key)
{
    switch (key)
    {
    case 27:
        running = false;
        break;

    case 'm':
    case 'M':
        ocl::setUseOpenCL(!cv::ocl::useOpenCL());
        cout << "Switched to " << (ocl::useOpenCL() ? "OpenCL enabled" : "CPU") << " mode\n";
        break;

    case 'g':
    case 'G':
        make_gray = !make_gray;
        cout << "Convert image to gray: " << (make_gray ? "YES" : "NO") << endl;
        break;

    case '1':
        scale *= 1.05;
        cout << "Scale: " << scale << endl;
        break;

    case 'q':
    case 'Q':
        scale /= 1.05;
        cout << "Scale: " << scale << endl;
        break;

    case '2':
        nlevels++;
        cout << "Levels number: " << nlevels << endl;
        break;

    case 'w':
    case 'W':
        nlevels = max(nlevels - 1, 1);
        cout << "Levels number: " << nlevels << endl;
        break;

    case '3':
        gr_threshold++;
        cout << "Group threshold: " << gr_threshold << endl;
        break;

    case 'e':
    case 'E':
        gr_threshold = max(0, gr_threshold - 1);
        cout << "Group threshold: " << gr_threshold << endl;
        break;

    case '4':
        hit_threshold += 0.25;
        cout << "Hit threshold: " << hit_threshold << endl;
        break;

    case 'r':
    case 'R':
        hit_threshold = max(0.0, hit_threshold - 0.25);
        cout << "Hit threshold: " << hit_threshold << endl;
        break;

    case 'c':
    case 'C':
        gamma_corr = !gamma_corr;
        cout << "Gamma correction: " << gamma_corr << endl;
        break;

    case 'o':
    case 'O':
        write_once = !write_once;
        break;
    }
}


inline void App::filterWorkBegin()
{
    filter_work_begin = getTickCount();
}

inline void App::filterWorkEnd()
{
    int64 delta = getTickCount() - filter_work_begin;
    double freq = getTickFrequency();
    filter_work_fps = freq / delta;
}

inline string App::filterWorkFps() const
{
    stringstream ss;
    ss << filter_work_fps;
    return ss.str();
}

inline void App::workBegin()
{
    work_begin = getTickCount();
}

inline void App::workEnd()
{
    int64 delta = getTickCount() - work_begin;
    double freq = getTickFrequency();
    work_fps = freq / delta;
}

inline string App::workFps() const
{
    stringstream ss;
    ss << work_fps;
    return ss.str();
}
