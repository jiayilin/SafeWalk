//
//  ViewController.mm
//  My_Camera_App
//
//  Created by 林佳艺 on 9/15/15.
//  Copyright © 2015 jiayi. All rights reserved.
//

#import "ViewController.h"
#import <AudioToolbox/AudioServices.h>

// Include stdlib.h and std namespace so we can mix C++ code in here
#include <stdlib.h>
using namespace std;

@interface ViewController()
{
    UIImageView *liveView_; // Live output from the camera
    UIImageView *resultView_; // Preview view of everything...
    UIImageView *imageView_;
    UIButton *takephotoButton_, *goliveButton_; // Button to initiate OpenCV processing of image
    CvPhotoCamera *photoCamera_; // OpenCV wrapper class to simplfy camera access through AVFoundation
}
@end

@implementation ViewController

const cv::Scalar RED = cvScalar(0,0,255);


//===============================================================================================
// Setup view for excuting App
- (void)viewDidLoad {
    [super viewDidLoad];
    
    // Do any additional setup after loading the view, typically from a nib.
    
    // 1. Setup the your OpenCV view, so it takes up the entire App screen......
    int view_width = self.view.frame.size.width;
    int view_height = (640*view_width)/480; // Work out the viw-height assuming 640x480 input
    int view_offset = (self.view.frame.size.height - view_height)/2;
    liveView_ = [[UIImageView alloc] initWithFrame:CGRectMake(0.0, view_offset, view_width, view_height)];
    [self.view addSubview:liveView_]; // Important: add liveView_ as a subview
    //resultView_ = [[UIImageView alloc] initWithFrame:CGRectMake(0.0, 0.0, 960, 1280)];
    resultView_ = [[UIImageView alloc] initWithFrame:CGRectMake(0.0, view_offset, view_width, view_height)];
    [self.view addSubview:resultView_]; // Important: add resultView_ as a subview
    resultView_.hidden = true; // Hide the view
    
    // 2. Setup a timmer to take a picture repeatedly
    [NSTimer scheduledTimerWithTimeInterval:2 target:self
                                   selector:@selector(takePicture) userInfo:nil repeats:YES];
    
    resultView_.contentMode = UIViewContentModeScaleAspectFit;
    
    // 3. Initialize the camera parameters and start the camera (inside the App)
    photoCamera_ = [[CvPhotoCamera alloc] initWithParentView:liveView_];
    photoCamera_.delegate = self;
    
    // This chooses whether we use the front or rear facing camera
    photoCamera_.defaultAVCaptureDevicePosition = AVCaptureDevicePositionBack;
    
    // This is used to set the image resolution
    photoCamera_.defaultAVCaptureSessionPreset = AVCaptureSessionPreset640x480;
    
    // This is used to determine the device orientation
    photoCamera_.defaultAVCaptureVideoOrientation = AVCaptureVideoOrientationPortrait;
    
    // This starts the camera capture
    [photoCamera_ start];
    
}

//===============================================================================================
// This member function is executed after certain time interval has pressed
- (void)takePicture {
    [photoCamera_ takePicture];
}

//===============================================================================================
// To be compliant with the CvPhotoCameraDelegate we need to implement these two methods
- (void)photoCamera:(CvPhotoCamera *)photoCamera capturedImage:(UIImage *)image
{
    [photoCamera_ stop];
    resultView_.hidden = false; // Turn the hidden view on
    
    //implement OpenCV code here to process images.
    
    cv::Mat cvImage = [self cvMatFromUIImage:image];
    cv::transpose(cvImage, cvImage);
    cv::Mat gray; cv::cvtColor(cvImage, gray, CV_RGBA2GRAY); // Convert to grayscale
    //cv::transpose(gray, gray);
    cout<<gray.size()<<endl;
    
    //resultView_.image = [self UIImageFromCVMat:gray];
    
    int width = gray.cols;
    int height = gray.rows;
    cv::Mat upper = gray(cv::Rect(0, 0, width, int(height/2)));
    cv::Mat lower = gray(cv::Rect(0, int(height/2), width, int(height/2)));
    //cout<<upper<<endl;
    
    cv::Mat display_im1; cv::cvtColor(upper, display_im1, CV_GRAY2BGR);
    cv::Mat display_im2; cv::cvtColor(lower, display_im2, CV_GRAY2BGR);
    
    NSDate *methodStart = [NSDate date];
    cv::Mat display_im_gray = [self calcDiff_gray:gray];
    //cv::Mat display_im_color = [self calcDiff_color:cvImage];
    NSDate *methodFinish = [NSDate date];
    NSTimeInterval executionTime = [methodFinish timeIntervalSinceDate:methodStart];
    NSLog(@"executionTime = %f", executionTime);
    
    resultView_.image = [self UIImageFromCVMat:display_im_gray];
    // Special part to ensure the image is rotated properly when the image is converted back
    //UIImage *resImage = MatToUIImage(display_im1);
    /*resultView_.image =  [UIImage imageWithCGImage:[resImage CGImage]
                                             scale:1.0
                                       orientation: UIImageOrientationLeftMirrored];
    */
    // After processing the captured photo, start the photoCamera and get more photos
    cout << "finish processing" << endl;
    resultView_.hidden = false; // Hide the result view again
    [photoCamera_ start];
    
}


- (cv::Mat) calcDiff_color: (cv::Mat) img {
    //srand ( time(NULL) );
    int width = img.cols;
    int height = img.rows;
    cv::Mat upper = img(cv::Rect(0, 0, width, int(height/2)));
    cv::Mat lower = img(cv::Rect(0, int(height/2), width, int(height/2)));
    //cout<<upper<<endl;
    
    cv::Mat display_im1 = upper; //cv::cvtColor(upper, display_im1, CV_GRAY2BGR);
    cv::Mat display_im2 = lower; //cv::cvtColor(lower, display_im2, CV_GRAY2BGR);
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
    //cout<<"randx: "<<endl<<randx<<endl;
    
    
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
            for (int k=0; k<3; k++) {
                double avg1 = (int)upper.at<cv::Vec3b>(py[0][j], px).val[k]
                + (int)upper.at<cv::Vec3b>(py[0][j]-1, px-1).val[k]
                + (int)upper.at<cv::Vec3b>(py[0][j], px-1).val[k]
                + (int)upper.at<cv::Vec3b>(py[0][j]+1, px-1).val[k]
                + (int)upper.at<cv::Vec3b>(py[0][j]-1, px).val[k]
                + (int)upper.at<cv::Vec3b>(py[0][j]+1, px).val[k]
                + (int)upper.at<cv::Vec3b>(py[0][j]-1, px+1).val[k]
                + (int)upper.at<cv::Vec3b>(py[0][j], px+1).val[k]
                + (int)upper.at<cv::Vec3b>(py[0][j]+1, px+1).val[k];
                double avg2 = (int)lower.at<cv::Vec3b>(py[1][j], px).val[k]
                + (int)lower.at<cv::Vec3b>(py[1][j]-1, px-1).val[k]
                + (int)lower.at<cv::Vec3b>(py[1][j], px-1).val[k]
                + (int)lower.at<cv::Vec3b>(py[1][j]+1, px-1).val[k]
                + (int)lower.at<cv::Vec3b>(py[1][j]-1, px).val[k]
                + (int)lower.at<cv::Vec3b>(py[1][j]+1, px).val[k]
                + (int)lower.at<cv::Vec3b>(py[1][j]-1, px+1).val[k]
                + (int)lower.at<cv::Vec3b>(py[1][j], px+1).val[k]
                + (int)lower.at<cv::Vec3b>(py[1][j]+1, px+1).val[k];
                avg1 /= 9;
                avg2 /= 9;
                //cout<<"avg1: "<<avg1<<", avg2: "<<avg2<<endl;
                diff += (avg1 - avg2) * (avg1 - avg2);
                count++;
            }
        }
    }
    diff = sqrt(diff / (numValid*3*3));
    cout<<"diff color: "<<diff<<endl;
    cout<<"# comps: "<<count<<", # valid points: "<<(numValid*3*3)<<endl<<endl;
    
    
    for (int i=0; i<numPoint; i++) {
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
    
    return complete;
    
}




- (cv::Mat) calcDiff_gray: (cv::Mat) gray {
    //srand ( time(NULL) );
    int width = gray.cols;
    int height = gray.rows;
    cv::Mat upper = gray(cv::Rect(0, 0, width, int(height/2)));
    cv::Mat lower = gray(cv::Rect(0, int(height/2), width, int(height/2)));
    //cout<<upper<<endl;
    
    cv::Mat display_im1; cv::cvtColor(upper, display_im1, CV_GRAY2BGR);
    cv::Mat display_im2; cv::cvtColor(lower, display_im2, CV_GRAY2BGR);
    cout<<"lower half dimensions: "<<lower.size()<<endl;
    
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
    //cout<<"randx: "<<endl<<randx<<endl;
    
    
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
    cout<<"# comps: "<<count<<", # valid points: "<<(numValid*3)<<endl<<endl;
    
    if (diff > 50){
        AudioServicesPlayAlertSound(kSystemSoundID_Vibrate);
        AudioServicesPlaySystemSound(kSystemSoundID_Vibrate);
    }
    
    
    for (int i=0; i<numPoint; i++) {
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
    
    return complete;
    
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
