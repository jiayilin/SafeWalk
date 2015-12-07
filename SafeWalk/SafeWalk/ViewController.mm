//
//  ViewController.m
//  SafeWalk
//
//  Created by Yangming Chong, Jiayi Lin on 11/21/15.
//  Copyright © 2015 jiayi. All rights reserved.
//
#import <GPUImage/GPUImage.h>
#import "ViewController.h"
#ifdef __cplusplus
#include <opencv2/opencv.hpp> // Includes the opencv library
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <stdlib.h> // Include the standard library
#include <iostream>  // for cout, getline
#include <sstream>  // for istringstream
#include <string>   // for string
#include <fstream>
#include <algorithm>
#endif

using namespace std;
using namespace cv;


@interface ViewController () {
    // Setup the view
    UIImageView *imageView_;
}
@end

@implementation ViewController


- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    
    // Read in the image
    UIImage *image = [UIImage imageNamed:@"wall.png"]; // new_view.jpg and prince_book.jpg
    //UIImage *image = [UIImage imageNamed:@"shadow.png"];
    if(image == nil) cout << "Cannot read in the file !!" << endl;
    
    // Setup the display
    // Setup the your imageView_ view, so it takes up the entire App screen......
    imageView_ = [[UIImageView alloc] initWithFrame:CGRectMake(0.0, 0.0, self.view.frame.size.width, self.view.frame.size.height)];
    // Important: add OpenCV_View as a subview
    [self.view addSubview:imageView_];
    
    // Ensure aspect ratio looks correct
    imageView_.contentMode = UIViewContentModeScaleAspectFit;
    
    // Another way to convert between cvMat and UIImage (using member functions)
    
    cv::Mat cvImage = [self cvMatFromUIImage:image];
    cv::Mat gray; cv::cvtColor(cvImage, gray, CV_RGBA2GRAY); // Convert to grayscale
    
    //cv::Mat blurred;
    //cv::GaussianBlur(gray, gray, cv::Size(27,27), 0, 0);
    
    int width = gray.cols;
    int height = gray.rows;
    cv::Mat upper = gray(cv::Rect(0, 0, width, int(height*3/5)));
    //cv::Mat lower = gray(cv::Rect(0, int(height/2), width, int(height/2)));
    //cout<<upper<<endl;
    
    /*cv::Mat display_im1; cv::cvtColor(upper, display_im1, CV_GRAY2BGR);
    cv::Mat display_im2; cv::cvtColor(lower, display_im2, CV_GRAY2BGR);
    cout<<"dimensions: "<<lower.size()<<endl;
    
    int numPoint = 10;
    cv::Mat randx = cv::Mat::zeros(1,numPoint,CV_64FC1);
    cv::Mat mean = cv::Mat::ones(1,1,CV_64FC1) * width / 2;
    cv::Mat sigma= cv::Mat::ones(1,1,CV_64FC1) * width / 4;
    cv::randn(randx,  mean, sigma);
    int py11 = height / 16;
    int py12 = height / 16 * 2;
    int py13 = height / 16 * 3;
    int py21 = height / 4 - height / 16;
    int py22 = height / 4;
    int py23 = height / 4 + height / 16;
    cout<<"randx: "<<endl<<randx<<endl;
    
    
    double diff = 0;
    int numValid = numPoint;
    for (int i=0; i<numPoint; i++) {
        if (randx.at<float>(0,i) <= 0 || randx.at<float>(0,i) >= width - 1) {
            numValid--;
            continue;
        }
        //cout<<"py1: "<<py1<<endl;
        cout<<"randx i: "<<(int)randx.at<double>(0,i)<<endl;
        //cout<<"randx i: "<<(int)upper.at<uchar>(py1, (int)randx.at<double>(0,i))<<endl;
        //double avg1 = upper.at<int>(py1, (int)randx.at<double>(0,i));
        
        int px = randx.at<double>(0,i);
        double avg1 = (int)upper.at<uchar>(py11, px)
                 + (int)upper.at<uchar>(py11-1, px-1)
                 + (int)upper.at<uchar>(py11, px-1)
                 + (int)upper.at<uchar>(py11+1, px-1)
                 + (int)upper.at<uchar>(py11-1, px)
                 + (int)upper.at<uchar>(py11+1, px)
                 + (int)upper.at<uchar>(py11-1, px+1)
                 + (int)upper.at<uchar>(py11, px+1)
                 + (int)upper.at<uchar>(py11+1, px+1);
        double avg2 = (int)lower.at<uchar>(py21, px)
                 + (int)lower.at<uchar>(py21-1, px-1)
                 + (int)lower.at<uchar>(py21, px-1)
                 + (int)lower.at<uchar>(py21+1, px-1)
                 + (int)lower.at<uchar>(py21-1, px)
                 + (int)lower.at<uchar>(py21+1, px)
                 + (int)lower.at<uchar>(py21-1, px+1)
                 + (int)lower.at<uchar>(py21, px+1)
                 + (int)lower.at<uchar>(py21+1, px+1);
        avg1 /= 9;
        avg2 /= 9;
        diff += (avg1 - avg2) * (avg1 - avg2);
        
        avg1 = (int)upper.at<uchar>(py12, px)
        + (int)upper.at<uchar>(py12-1, px-1)
        + (int)upper.at<uchar>(py12, px-1)
        + (int)upper.at<uchar>(py12+1, px-1)
        + (int)upper.at<uchar>(py12-1, px)
        + (int)upper.at<uchar>(py12+1, px)
        + (int)upper.at<uchar>(py12-1, px+1)
        + (int)upper.at<uchar>(py12, px+1)
        + (int)upper.at<uchar>(py12+1, px+1);
        avg2 = (int)lower.at<uchar>(py22, px)
        + (int)lower.at<uchar>(py22-1, px-1)
        + (int)lower.at<uchar>(py22, px-1)
        + (int)lower.at<uchar>(py22+1, px-1)
        + (int)lower.at<uchar>(py22-1, px)
        + (int)lower.at<uchar>(py22+1, px)
        + (int)lower.at<uchar>(py22-1, px+1)
        + (int)lower.at<uchar>(py22, px+1)
        + (int)lower.at<uchar>(py22+1, px+1);
        avg1 /= 9;
        avg2 /= 9;
        diff += (avg1 - avg2) * (avg1 - avg2);

        avg1 = (int)upper.at<uchar>(py13, px)
        + (int)upper.at<uchar>(py13-1, px-1)
        + (int)upper.at<uchar>(py13, px-1)
        + (int)upper.at<uchar>(py13+1, px-1)
        + (int)upper.at<uchar>(py13-1, px)
        + (int)upper.at<uchar>(py13+1, px)
        + (int)upper.at<uchar>(py13-1, px+1)
        + (int)upper.at<uchar>(py13, px+1)
        + (int)upper.at<uchar>(py13+1, px+1);
        avg2 = (int)lower.at<uchar>(py23, px)
        + (int)lower.at<uchar>(py23-1, px-1)
        + (int)lower.at<uchar>(py23, px-1)
        + (int)lower.at<uchar>(py23+1, px-1)
        + (int)lower.at<uchar>(py23-1, px)
        + (int)lower.at<uchar>(py23+1, px)
        + (int)lower.at<uchar>(py23-1, px+1)
        + (int)lower.at<uchar>(py23, px+1)
        + (int)lower.at<uchar>(py23+1, px+1);
        avg1 /= 9;
        avg2 /= 9;
        diff += (avg1 - avg2) * (avg1 - avg2);
    }
    diff = sqrt(diff) / (numValid*3);
    cout<<"diff: "<<diff<<endl;
    
    for (int i=0; i<numPoint; i++) {
        cv::Point pt;
        pt.x = randx.at<double>(0,i);
        pt.y = py11;
        cv::circle(display_im1, pt, 10, Scalar(255,0,0), 3);
        
        pt.y = py12;
        cv::circle(display_im1, pt, 10, Scalar(255,0,0), 3);
        
        pt.y = py13;
        cv::circle(display_im1, pt, 10, Scalar(255,0,0), 3);
    }*/
    
    /*vector<cv::KeyPoint> kp;
    
    int nfeatures = 1000;
    int edgeThresh = 200;
    cv::ORB orb(nfeatures, 1.2f, 8, edgeThresh);
    cv::Mat orbDes;
    orb(gray, cv::Mat(), kp, orbDes);
    cout<<"orbDes size: "<<orbDes.size()<<endl;*/
    
    
    // calculate standard variance of x and y of keypoints
    /*double mean_x = 0, mean_y = 0, var_x = 0, var_y = 0;
    for (int i=0; i<kp.size(); i++) {
        mean_x += kp[i].pt.x;
        mean_y += kp[i].pt.y;
    }
    mean_x /= kp.size();
    mean_y /= kp.size();
    for (int i=0; i<kp.size(); i++) {
        var_x += (kp[i].pt.x - mean_x) * (kp[i].pt.x - mean_x);
        var_y += (kp[i].pt.y - mean_y) * (kp[i].pt.y - mean_y);
    }
    var_x /= (kp.size()-1);
    var_y /= (kp.size()-1);
    cout<<"std x: "<<sqrt(var_x)<<endl;
    cout<<"std y: "<<sqrt(var_y)<<endl;*/
    
    
    // calculate the histogram of distribution of x and y of keypoints
    /*int numBin = 10;
    vector<int> count_x(numBin), count_y(numBin);
    for (int i=0; i<numBin; i++) {
        count_x[i] = count_y[i] = 0;
    }
    for (int i = 0; i<kp.size(); i++) {
        count_x[(int)(kp[i].pt.x * numBin / gray.cols)]++;
        count_y[(int)(kp[i].pt.y * numBin / gray.rows)]++;
    }
    cout<<"count_x: "<<endl;
    for (int i=0; i<numBin; i++) cout<<count_x[i]<<",";
    cout<<endl<<"count_x max: "<<*std::max_element(std::begin(count_x), std::end(count_x));
    cout<<endl<<"count_y: "<<endl;
    for (int i=0; i<numBin; i++) cout<<count_y[i]<<",";
    cout<<endl<<"count_y max: "<<*std::max_element(std::begin(count_y), std::end(count_y));
    
    
    
    
    
    
    for (int i=0; i<gray.rows; i++) {
        for (int j=0; j<gray.cols; j++) {
            gray.at<uchar>(i,j) = gray.at<uchar>(i,j) * 0.2;
        }
    }
    
    cv::Mat display_im1; cv::cvtColor(gray, display_im1, CV_GRAY2BGR);
    cv::drawKeypoints(display_im1, kp, display_im1);*/
    
    
    
    NSDate *methodStart = [NSDate date];
    cv::Mat display_im_edge = [self extractEdges:upper];
    NSDate *methodFinish = [NSDate date];
    NSTimeInterval executionTime = [methodFinish timeIntervalSinceDate:methodStart];
    NSLog(@"executionTime = %f", executionTime);
    
    /*vector<cv::KeyPoint> kp1, kp2;
    
    int nfeatures = 2000;
    int edgeThresh = 200;
    cv::ORB orb(nfeatures, 1.2f, 8, edgeThresh);
    cv::Mat orbDes1, orbDes2;
    orb(upper, cv::Mat(), kp1, orbDes1);
    orb(lower, cv::Mat(), kp2, orbDes2);
    cout<<"orbDes1 size: "<<orbDes1.size()<<endl;
    
    cv::Mat orbDesAll;
    cv::vconcat(orbDes1, orbDes2, orbDesAll);
    orbDesAll.convertTo(orbDesAll, CV_32F);
    cout<<"orbDesAll size: "<<orbDesAll.size()<<endl;
    
    int dictSize = 50;
    cv::TermCriteria tc(CV_TERMCRIT_ITER, 100, 0.001);
    int retries = 1;
    int flags = KMEANS_PP_CENTERS;
    cv::BOWKMeansTrainer bowTrainer(dictSize, tc, retries, flags);
    //Mat dictionary =  bowTrainer.cluster(orbDesAll);
    Mat dictionary = Mat(20, 32, CV_64F, cvScalar(0.));
    Mat udictionary;
    //cv::transpose(udictionary, udictionary);
    
    NSString *str = [[NSBundle mainBundle]pathForResource:@"dictionary" ofType:@"txt"];
    const char *dictionaryName = [str UTF8String];
    ifstream file(dictionaryName);
    string line;
    int row = 0;
    int col = 0;
    while (getline(file, line)){
        istringstream stream(line);
        double x;
        col = 0;  // reset column counter
        while (stream >> x) {
            dictionary.at<double>(row,col) = x;
            col++;
        }
        row++;
    }
    
    
    std::cout<<"dictionary size: "<<dictionary.size()<<endl;
    //std::cout<<dictionary<<endl;
    
    dictionary.convertTo(udictionary, CV_8UC1);
    
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
    Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("ORB");
    BOWImgDescriptorExtractor bowDE(extractor, matcher);
    bowDE.setVocabulary(udictionary);
    
    Mat bowDescriptor1, bowDescriptor2;
    bowDE.compute(upper, kp1, bowDescriptor1);
    bowDE.compute(lower, kp2, bowDescriptor2);
    std::cout<<"kp1 size: "<<kp1.size()<<endl;
    std::cout<<"bowDescriptor1 size: "<<bowDescriptor1.size()<<endl;
    
    cout<<"sum: "<<sum(bowDescriptor1)<<endl;
    std::cout<<"bowDescriptor2 size: "<<bowDescriptor2.size()<<endl;
    
    double dist = cv::norm(bowDescriptor1, bowDescriptor2);
    cout<<"dist: "<<dist<<endl;
    
    
    cv::drawKeypoints(display_im1, kp1, display_im1);
    cv::drawKeypoints(display_im2, kp2, display_im2);
    std::cout<<"number of key points: " << kp1.size()<< "," << kp2.size()<<endl;*/
    
    // Finally setup the view to display
    imageView_.image = [self UIImageFromCVMat:display_im_edge];
    
}

- (cv::Mat) extractEdges: (cv::Mat) gray {
    
    cout<<"dimension: "<<gray.size()<<endl;
    int lowThreshold = 30;
    int ratio = 3;
    int kernel_size = 3;
    cv::Mat detected_edges, resize_im, dst, cdst;
    
    // resize the image
    cv::resize(gray, resize_im, cv::Size(), 0.2, 0.2, cv::INTER_CUBIC);
    
    // Use Gaussian Blur to blur the image
    cv::GaussianBlur(resize_im, detected_edges, cv::Size(7,7), 0, 0);
    
    // Apply Canny function
    cv::Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );
    
    // Resize back
    cv::resize(detected_edges, detected_edges, gray.size(), 0, 0, cv::INTER_CUBIC);
    
    
    cvtColor(detected_edges, cdst, CV_GRAY2BGR);
    
    // Apply Hough Lines
    /*cv::vector<cv::Vec2f> lines;
    cv::HoughLines(detected_edges, lines, 1, CV_PI/180, 800, 0, 0 );
    
    // Draw lines
    for( size_t i = 0; i < lines.size(); i++ )
    {
        float rho = lines[i][0], theta = lines[i][1];
        cv::Point pt1, pt2;
        
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        pt1.x = cvRound(x0 + 5000*(-b));
        pt1.y = cvRound(y0 + 5000*(a));
        pt2.x = cvRound(x0 - 5000*(-b));
        pt2.y = cvRound(y0 - 5000*(a));
        cv::line( cdst, pt1, pt2, cv::Scalar(0,0,255), 3, CV_AA);
    }*/
    
    // Probablistic Hough
    int resizeFactor = 1;
    cv::vector<cv::Vec4i> lines;
    cv::HoughLinesP(detected_edges, lines, 1, CV_PI/180, 200, 300, 10);
    for( size_t i = 0; i < lines.size(); i++ )
    {
        cv::Vec4i l = lines[i];
        cv::line(cdst, cv::Point(l[0]/resizeFactor, l[1]/resizeFactor), cv::Point(l[2]/resizeFactor, l[3]/resizeFactor), cv::Scalar(0,0,255), 3, CV_AA);
    }
    
    
    
    return cdst;

}


- (cv::Mat) extractEdgesGPU: (cv::Mat) gray {
    
    // tranform to UIImage
    UIImage *inputImage = [self UIImageFromCVMat:gray];
    
    
    // Initialize filters
    // Grescale filter
    GPUImagePicture *grayImageSource = [[GPUImagePicture alloc] initWithImage:inputImage];
    GPUImageGrayscaleFilter *grayImageFilter = [[GPUImageGrayscaleFilter alloc] init];
    
    // Use transform to resize the image
    GPUImageTransformFilter *resizeFilter = [[GPUImageTransformFilter alloc] init];
    CGAffineTransform transform = CGAffineTransformMakeScale(0.2, 0.2);
    resizeFilter.affineTransform = transform;
    
    // Gaussian blur filter
    GPUImageGaussianBlurFilter *blurFilter = [[GPUImageGaussianBlurFilter alloc] init];
    blurFilter.blurRadiusInPixels = 2;
    blurFilter.blurPasses = 1;
    blurFilter.texelSpacingMultiplier = 1.1;
    
    // Canny edge detection filter
    GPUImageCannyEdgeDetectionFilter *cannyFilter = [[GPUImageCannyEdgeDetectionFilter alloc] init];
    cannyFilter.upperThreshold = 0.2;
    cannyFilter.lowerThreshold = 0.08;
    //cannyFilter.blurRadiusInPixels = 1;
    
    // Use transform to resize the image back
    GPUImageTransformFilter *backFilter = [[GPUImageTransformFilter alloc] init];
    CGAffineTransform transformback = CGAffineTransformMakeScale(5, 5);
    backFilter.affineTransform = transformback;
    
    // Daisy chain the filters together (you can add as many filters as you like)
    [grayImageSource addTarget:grayImageFilter];
    [grayImageFilter addTarget:resizeFilter];
    [resizeFilter addTarget:blurFilter];
    [blurFilter addTarget:cannyFilter];
    [cannyFilter addTarget:backFilter];
    [backFilter useNextFrameForImageCapture];
    //[cannyFilter useNextFrameForImageCapture];
    
    // Process the image
    [grayImageSource processImage];
    
    //UIImage *detected_edges_ui = [backFilter imageFromCurrentFramebuffer];
    UIImage *detected_edges_ui = [backFilter imageFromCurrentFramebuffer];
    
    //cv::Mat detected_edges = [self cvMatFromUIImage:detected_edges_ui];
    cv::Mat cdst = [self cvMatFromUIImage:detected_edges_ui];
    cout<<"cdst channels: "<<cdst.channels()<<endl;
    cout<<"cdst dimension: "<<cdst.size()<<endl;
    
    //cv::Mat cdst;
    // Apply Hough Lines
    //cvtColor(detected_edges, cdst, CV_GRAY2BGR);
    
    return cdst;
    
    cv::Mat detected_edges;
    cvtColor(cdst, detected_edges, CV_RGBA2GRAY);
    
    cout<<"cvtcolor finished"<<endl;
    
    cv::vector<cv::Vec2f> lines;
    cv::HoughLines(detected_edges, lines, 1, CV_PI/180, 400, 0, 0 );
    
    
    cout<<"hough finished"<<endl;
    
    const cv::Scalar RED = cv::Scalar(0,0,255); // Set the RED color
    
    // Draw lines
    for( size_t i = 0; i < lines.size(); i++ )
    {
        float rho = lines[i][0], theta = lines[i][1];
        cv::Point pt1, pt2;
        
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        pt1.x = cvRound(x0 + 5000*(-b));
        pt1.y = cvRound(y0 + 5000*(a));
        pt2.x = cvRound(x0 - 5000*(-b));
        pt2.y = cvRound(y0 - 5000*(a));
        cv::line( cdst, pt1, pt2, RED, 3, CV_AA);
    }
    return cdst;
}



- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

// Member functions for converting from cvMat to UIImage
- (cv::Mat)cvMatFromUIImage:(UIImage *)image
{
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
    CGFloat cols = image.size.width;
    CGFloat rows = image.size.height;
    
    cv::Mat cvMat(rows, cols, CV_8UC4); // 8 bits per component, 4 channels (color channels + alpha)
    
    CGContextRef contextRef = CGBitmapContextCreate(cvMat.data,                 // Pointer to  data
                                                    cols,                       // Width of bitmap
                                                    rows,                       // Height of bitmap
                                                    8,                          // Bits per component
                                                    cvMat.step[0],              // Bytes per row
                                                    colorSpace,                 // Colorspace
                                                    kCGImageAlphaNoneSkipLast |
                                                    kCGBitmapByteOrderDefault); // Bitmap info flags
    
    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), image.CGImage);
    CGContextRelease(contextRef);
    
    return cvMat;
}
// Member functions for converting from UIImage to cvMat
-(UIImage *)UIImageFromCVMat:(cv::Mat)cvMat
{
    NSData *data = [NSData dataWithBytes:cvMat.data length:cvMat.elemSize()*cvMat.total()];
    CGColorSpaceRef colorSpace;
    
    if (cvMat.elemSize() == 1) {
        colorSpace = CGColorSpaceCreateDeviceGray();
    } else {
        colorSpace = CGColorSpaceCreateDeviceRGB();
    }
    
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    
    // Creating CGImage from cv::Mat
    CGImageRef imageRef = CGImageCreate(cvMat.cols,                                 //width
                                        cvMat.rows,                                 //height
                                        8,                                          //bits per component
                                        8 * cvMat.elemSize(),                       //bits per pixel
                                        cvMat.step[0],                            //bytesPerRow
                                        colorSpace,                                 //colorspace
                                        kCGImageAlphaNone|kCGBitmapByteOrderDefault,// bitmap info
                                        provider,                                   //CGDataProviderRef
                                        NULL,                                       //decode
                                        false,                                      //should interpolate
                                        kCGRenderingIntentDefault                   //intent
                                        );
    
    
    // Getting UIImage from CGImage
    UIImage *finalImage = [UIImage imageWithCGImage:imageRef];
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    
    return finalImage;
}

@end
