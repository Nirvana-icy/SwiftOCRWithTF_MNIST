//
//  XGDigitalRecognizeService.h
//  SwiftOCR Camera
//
//  Created by JinglongBi on 2017/7/21.
//  Copyright © 2017年 Nicolas Camenisch. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

@interface XGDigitalRecognizeService : NSObject

-(NSInteger)recognizeDigitalWithImg:(UIImage *)orignalImage;

@end
