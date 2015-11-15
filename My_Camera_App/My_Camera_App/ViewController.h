//
//  ViewController.h
//  My_Camera_App
//
//  Created by 林佳艺 on 9/15/15.
//  Copyright © 2015 jiayi. All rights reserved.
//

#import <UIKit/UIKit.h>

#ifdef __cplusplus
#import <opencv2/opencv.hpp>
#import "opencv2/highgui/ios.h"
#endif

@interface ViewController : UIViewController<CvPhotoCameraDelegate>

@end

