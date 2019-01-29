#include <iostream>
#include <stdio.h>

#include <pangolin/pangolin.h>
#include <pangolin/gl/glcuda.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "kernel.h"

#define USE_TEXTURE_CUDA_ARRAY

// Host code
int main()
{
    initCuda();

    // Create OpenGL window in single line
    pangolin::CreateWindowAndBind("Main",640,480);

    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);
    
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(640,480,420,420,320,240,0.1,1000),
        pangolin::ModelViewLookAt(-1,1,-1, 0,0,0, pangolin::AxisY)
    );

    // Aspect ratio allows us to constrain width and height whilst fitting within specified
    // bounds. A positive aspect ratio makes a view 'shrink to fit' (introducing empty bars),
    // whilst a negative ratio makes the view 'grow to fit' (cropping the view).
    pangolin::View& d_cam = pangolin::Display("cam")
        .SetBounds(0,1.0f,0,1.0f,-640/480.0)
        .SetHandler(new pangolin::Handler3D(s_cam));

    // This view will take up no more than a third of the windows width or height, and it
    // will have a fixed aspect ratio to match the image that it will display. When fitting
    // within the specified bounds, push to the top-left (as specified by SetLock).
    pangolin::View& d_image = pangolin::Display("image")
        .SetBounds(0.f, 1.f, 0.f, 1.f,640.0/480)
        .SetLock(pangolin::LockLeft, pangolin::LockTop);

    cv::Mat img = cv::imread("../data/ye_high2.png");   
    if(img.empty()) 
        throw std::runtime_error("Could not open image");

    //if(img.type() != CV_8UC3)
    //    img.convertTo(img, CV_8UC3);

    int width = img.cols;
    int height = img.rows;   


    // Copy to host data without alignment issues, ...
    // Could also just do a cv::cvtColor(..., CV_BGR2BGRA)
    uchar4* h_data = new uchar4[width * height];
    for (int i = 0; i < img.total(); i++)
    {
        auto vec = img.at<cv::Vec3b>(i);
        h_data[i] = make_uchar4(vec[0], vec[1], vec[2], 1);
    }   

    // Temporary array where we safe our output data to
    uchar4* tmp;
    cudaSafeCall(cudaMalloc(&tmp, width * height * sizeof(uchar4)));

#ifdef USE_TEXTURE_CUDA_ARRAY

    pangolin::GlTextureCudaArray cudaTexture(width, height, GL_RGBA, true, 0, GL_BGRA, GL_UNSIGNED_BYTE);
    pangolin::CudaScopedMappedArray cudaTextureArrayMapping(cudaTexture);
    cudaArray_t inputArray = *cudaTextureArrayMapping;

#else
   
    // Register array with texture *before* filling array...
    pangolin::GlTexture imageTexture(width, height, GL_RGBA, false, 0, GL_BGRA, GL_UNSIGNED_BYTE);

    cudaGraphicsResource_t resource;
    cudaSafeCall(cudaGraphicsGLRegisterImage(&resource, imageTexture.tid, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));

    // Map and bind
    cudaArray_t inputArray;
    cudaSafeCall(cudaGraphicsMapResources(1, &resource, 0));
    cudaSafeCall(cudaGraphicsSubResourceGetMappedArray(&inputArray, resource, 0, 0));
    
    cudaSafeCall(cudaGraphicsUnmapResources(1, &resource, 0));
#endif  

    // Copy host data to array
    cudaSafeCall(cudaMemcpyToArray(inputArray, 0, 0, h_data, width * height * sizeof(h_data[0]), cudaMemcpyHostToDevice));

    // Default hooks for exiting (Esc) and fullscreen (tab).
    while(!pangolin::ShouldQuit())
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);

        // Transfrom image
        transformImage<uchar4>(inputArray, tmp, width, height);
        
        // Copy the transformed image back into the inputArray
        cudaSafeCall(cudaMemcpyToArray(inputArray, 0, 0, tmp, width * height * sizeof(tmp[0]), cudaMemcpyDeviceToDevice));

        // Display the image
        d_image.Activate();
        glColor3f(1.0,1.0,1.0);
        cudaTexture.RenderToViewportFlipY();

        pangolin::FinishFrame();
    }

    cudaSafeCall(cudaFree(tmp));
    
    return 0;
}