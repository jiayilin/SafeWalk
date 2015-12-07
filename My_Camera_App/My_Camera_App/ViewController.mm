//
//  ViewController.mm
//  My_Camera_App
//
//  Created by 林佳艺 on 9/15/15.
//  Copyright © 2015 jiayi. All rights reserved.
//

#import "ViewController.h"
#import <AudioToolbox/AudioServices.h>
#import <GPUImage/GPUImage.h>

// Include stdlib.h and std namespace so we can mix C++ code in here
#include <stdlib.h>
using namespace std;

@interface ViewController()
{
    UIImageView *liveView_; // Live output from the camera
    UIImageView *resultView_; // Preview view of everything...
    UIImageView *imageView_;
    CvVideoCamera *videoCamera;
}
@end

@implementation ViewController

const cv::Scalar RED = cvScalar(0,0,255);


//===============================================================================================
// Setup view for excuting App
- (void)viewDidLoad {
    [super viewDidLoad];
    
    // Do any additional setup after loading the view, typically from a nib.
    
    //Setup the your OpenCV view, so it takes up the entire App screen......
    int view_width = self.view.frame.size.width;
    int view_height = (640*view_width)/480; // Work out the viw-height assuming 640x480 input
    //int view_height = (1920*view_width)/1080;
    int view_offset = (self.view.frame.size.height - view_height)/2;
    liveView_ = [[UIImageView alloc] initWithFrame:CGRectMake(0.0, view_offset, view_width, view_height)];
    [self.view addSubview:liveView_]; // Important: add liveView_ as a subview
    //resultView_ = [[UIImageView alloc] initWithFrame:CGRectMake(0.0, 0.0, 960, 1280)];
    resultView_ = [[UIImageView alloc] initWithFrame:CGRectMake(0.0, view_offset, view_width, view_height)];
    [self.view addSubview:resultView_]; // Important: add resultView_ as a subview
    resultView_.hidden = true; // Hide the view
    
    resultView_.contentMode = UIViewContentModeScaleAspectFit;
    
    // Initialize the video camera
    videoCamera = [[CvVideoCamera alloc] initWithParentView:liveView_];
    videoCamera.delegate = self;
    
    videoCamera.defaultAVCaptureDevicePosition = AVCaptureDevicePositionBack;
    videoCamera.defaultAVCaptureSessionPreset = AVCaptureSessionPreset640x480;
    //videoCamera.defaultAVCaptureSessionPreset = AVCaptureSessionPreset1920x1080;
    videoCamera.defaultAVCaptureVideoOrientation = AVCaptureVideoOrientationPortrait;
    videoCamera.defaultFPS = 30;
    videoCamera.rotateVideo = YES;
    
    [videoCamera start];
}

int i;


- (void)processImage:(cv::Mat&)image;
{
    i++;
    // process the image every 3 frame
    if (i==10) {
        resultView_.hidden = false; // Turn the hidden view on
        
        //implement OpenCV code here to process images.
        cv::Mat sharpenImage;
        cv::Mat cvImage = image;//[self cvMatFromUIImage:image];
        
        //cv::GaussianBlur(cvImage, sharpenImage, cv::Size(0,0), 23);
        //cv::addWeighted(cvImage, 2, sharpenImage, -1, 0, sharpenImage);
        
        cv::Mat gray; cv::cvtColor(cvImage, gray, CV_RGBA2GRAY); // Convert to grayscale
        //cout<<gray.size()<<endl;
        
        //resultView_.image = [self UIImageFromCVMat:gray];
        
        //int width = gray.cols;
        //int height = gray.rows;
        //cv::Mat upper = gray(cv::Rect(0, 0, width, int(height/2)));
        //cv::Mat lower = gray(cv::Rect(0, int(height/2), width, int(height/2)));
        //cout<<upper<<endl;
        
        //cv::Mat display_im1; cv::cvtColor(upper, display_im1, CV_GRAY2BGR);
        //cv::Mat display_im2; cv::cvtColor(lower, display_im2, CV_GRAY2BGR);
        
        NSDate *methodStart = [NSDate date];
        
        bool ifObstacle_Diff = [self calcDiff_gray:gray];
        if (ifObstacle_Diff) {
            AudioServicesPlaySystemSound(1008);
        }
        else {
            bool ifObstacle_ORB = [self ORBdistribution:gray];
            if (ifObstacle_ORB) {
                AudioServicesPlaySystemSound(1007);
            }
        }
        
        
        // record execution time
        NSDate *methodFinish = [NSDate date];
        NSTimeInterval executionTime = [methodFinish timeIntervalSinceDate:methodStart];
        NSLog(@"executionTime = %f", executionTime);
        cout<<endl;
        
        //cv::Mat display_im1; cv::cvtColor(gray, display_im1, CV_GRAY2BGR);
        

        i = 0;
        resultView_.hidden = true; // Hide the result view again
    }
}

- (bool) ORBdistribution: (cv::Mat) gray {
    double ORBratio = 0.3;
    
    int nfeatures = 1000;
    int edgeThresh = 50;
    
    vector<cv::KeyPoint> kp;
    cv::OrbFeatureDetector detector(nfeatures, 1.2f, 8, edgeThresh, 0);
    detector.detect(gray, kp);
    cout<<"kp size: "<<kp.size()<<endl;
    
    if (kp.size()>60) {
        int numBin = 10;
        vector<int> count_x(numBin), count_y(numBin);
        for (int i=0; i<numBin; i++) {
            count_x[i] = count_y[i] = 0;
        }
        for (int i = 0; i<kp.size(); i++) {
            count_x[(int)(kp[i].pt.x * numBin / gray.cols)]++;
            count_y[(int)(kp[i].pt.y * numBin / gray.rows)]++;
        }
        //cout<<"count_x: ";
        //for (int i=0; i<numBin; i++) cout<<count_x[i]<<",";
        
        std::vector<int> range_x(count_x.begin() + 2, count_x.end() - 2);
        std::vector<int> range_y(count_y.begin(), count_y.end() - 5);
        
        double max_x = (double)*std::max_element(std::begin(range_x), std::end(range_x)) / kp.size();
        double max_y = (double)*std::max_element(std::begin(range_y), std::end(range_y)) / kp.size();
        cout<<"max_x: "<<max_x<<endl;
        
        //cout<<"count_y: ";
        //for (int i=0; i<numBin; i++) cout<<count_y[i]<<",";
        cout<<"max_y: "<<max_y<<endl;
        
        if (max_x > ORBratio || max_y > ORBratio) {
            return true;
        }
    }
    
    return false;
}




- (bool) extractEdges: (cv::Mat) gray {
    return false;
 }



- (bool) calcDiff_gray: (cv::Mat) gray {
    double alarm_thresh = 60;
    
    int width = gray.cols;
    int height = gray.rows;
    cv::Mat upper = gray(cv::Rect(0, 0, width, int(height/2)));
    cv::Mat lower = gray(cv::Rect(0, int(height/2), width, int(height/2)));
    //cout<<upper<<endl;
    
    cv::Mat display_im1; cv::cvtColor(upper, display_im1, CV_GRAY2BGR);
    cv::Mat display_im2; cv::cvtColor(lower, display_im2, CV_GRAY2BGR);
    //cout<<"lower half dimensions: "<<lower.size()<<endl;
    
    int numPoint = 10;
    cv::Mat randx = cv::Mat::zeros(1,numPoint,CV_64FC1);
    cv::Mat mean = cv::Mat::ones(1,1,CV_64FC1) * width / 2;
    cv::Mat sigma= cv::Mat::ones(1,1,CV_64FC1) * width / 4;
    cv::randn(randx,  mean, sigma);
    int py[2][3];
    py[0][0] = height / 16;
    py[0][1] = height / 16 * 2;
    py[0][2] = height / 16 * 3;
    py[1][0] = height / 4 - height / 16;
    py[1][1] = height / 4;
    py[1][2] = height / 4 + height / 16;
    
    
    double diff = 0;
    int count = 0;
    int numValid = numPoint;
    for (int i=0; i<numPoint; i++) {
        if (randx.at<float>(0,i) <= 0 || randx.at<float>(0,i) >= width - 1) {
            numValid--;
            continue;
        }
        
        int px = randx.at<double>(0,i);
        
        for (int j=0; j<3; j++) {
            double avg1 = (int)upper.at<uchar>(py[0][j], px)
            + (int)upper.at<uchar>(py[0][j]-1, px-1)
            + (int)upper.at<uchar>(py[0][j], px-1)
            + (int)upper.at<uchar>(py[0][j]+1, px-1)
            + (int)upper.at<uchar>(py[0][j]-1, px)
            + (int)upper.at<uchar>(py[0][j]+1, px)
            + (int)upper.at<uchar>(py[0][j]-1, px+1)
            + (int)upper.at<uchar>(py[0][j], px+1)
            + (int)upper.at<uchar>(py[0][j]+1, px+1);
            double avg2 = (int)lower.at<uchar>(py[1][j], px)
            + (int)lower.at<uchar>(py[1][j]-1, px-1)
            + (int)lower.at<uchar>(py[1][j], px-1)
            + (int)lower.at<uchar>(py[1][j]+1, px-1)
            + (int)lower.at<uchar>(py[1][j]-1, px)
            + (int)lower.at<uchar>(py[1][j]+1, px)
            + (int)lower.at<uchar>(py[1][j]-1, px+1)
            + (int)lower.at<uchar>(py[1][j], px+1)
            + (int)lower.at<uchar>(py[1][j]+1, px+1);
            avg1 /= 9;
            avg2 /= 9;
            diff += (avg1 - avg2) * (avg1 - avg2);
            count++;
        }
    }
    diff = sqrt(diff / (numValid*3));
    cout<<"diff gray: "<<diff<<endl;
    //cout<<"# comps: "<<count<<", # valid points: "<<(numValid*3)<<endl<<endl;
    
    if (diff > alarm_thresh) return true;
    else return false;
    
    
    /*for (int i=0; i<numPoint; i++) {
        cv::Point pt;
        pt.x = randx.at<double>(0,i);
        
        for (int j=0; j<3; j++) {
            pt.y = py[0][j];
            cv::circle(display_im1, pt, 10, cv::Scalar(255,0,0), 3);
            
            pt.y = py[1][j];
            cv::circle(display_im2, pt, 10, cv::Scalar(255,0,0), 3);
        }
    }
    
    cv::Mat complete;
    cv::vconcat(display_im1, display_im2, complete);
    
    // Print out diff in the middle of the screen
    cv::Point text_origin(complete.size().width*0.5 ,complete.size().height*0.95);
    char d[10];
    sprintf(d, "diff %f",(double)diff);
    cv::putText(complete, d, text_origin, cv::FONT_HERSHEY_SIMPLEX, 0.8, RED);
    
    return complete;*/
    
}



- (void)photoCameraCancel:(CvPhotoCamera *)photoCamera
{
    
}
//===============================================================================================

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


//===============================================================================================
// Standard memory warning component added by Xcode
- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

@end
