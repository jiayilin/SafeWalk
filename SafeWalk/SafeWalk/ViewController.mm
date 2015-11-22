//
//  ViewController.m
//  SafeWalk
//
//  Created by Yangming Chong, Jiayi Lin on 11/21/15.
//  Copyright © 2015 jiayi. All rights reserved.
//

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
    UIImage *image = [UIImage imageNamed:@"brick.jpg"]; // new_view.jpg and prince_book.jpg
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
    //cv::GaussianBlur(gray, blurred, cv::Size(17,17), 0, 0);
    
    int width = gray.cols;
    int height = gray.rows;
    cv::Mat upper = gray(cv::Rect(0, 0, width, int(height/2)));
    cv::Mat lower = gray(cv::Rect(0, int(height/2), width, int(height/2)));
    cout<<"dimensions: "<<lower.size()<<endl;
    
    vector<cv::KeyPoint> kp1, kp2;
    
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
    
    cv::Mat display_im1; cv::cvtColor(upper, display_im1, CV_GRAY2BGR);
    cv::Mat display_im2; cv::cvtColor(lower, display_im2, CV_GRAY2BGR);
    cv::drawKeypoints(display_im1, kp1, display_im1);
    cv::drawKeypoints(display_im2, kp2, display_im2);
    std::cout<<"number of key points: " << kp1.size()<< "," << kp2.size()<<endl;
    
    // Finally setup the view to display
    imageView_.image = [self UIImageFromCVMat:display_im1];
    
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
