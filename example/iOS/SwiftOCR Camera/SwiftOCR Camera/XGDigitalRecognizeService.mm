//
//  XGDigitalRecognizeService.m
//  SwiftOCR Camera
//
//  Created by JinglongBi on 2017/7/21.
//  Copyright © 2017年 Nicolas Camenisch. All rights reserved.
//

#import "XGDigitalRecognizeService.h"

#import <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>

static constexpr int kImageSize = 28;
static constexpr int kOutputs = 10;
static constexpr int kInputLength = kImageSize * kImageSize;

@implementation XGDigitalRecognizeService
{
    tensorflow::GraphDef graph;
    tensorflow::Session *session;
}

-(instancetype)init {
    self = [super init];
    if (self) {
        // init TensorFlow
        NSString *path = [[NSBundle mainBundle] pathForResource:@"final" ofType:@"pb"];
        [self loadGraphFromPath:path];
        [self createSession];
    }
    return self;
}

-(NSInteger)recognizeDigitalWithImg:(UIImage *)orignalImage {
    // 1. 读取图片，将图片scale，读取像素做normalize
    UIImage *scaledImage = [self scaleImage:orignalImage];
    UIImage *image = [self convertImageToGrayScale:scaledImage];
    
    
    tensorflow::Tensor x(tensorflow::DT_FLOAT, tensorflow::TensorShape({1,kInputLength}));
    
    NSArray *pixel = [self getRGBAsFromImage:image atX:0 andY:0 count:kInputLength];
    
    for (auto i = 0; i < kInputLength; i++) {
        UIColor *color = pixel[i];
        CGFloat red = 0.0, green = 0.0, blue = 0.0, alpha =0.0;
        [color getRed:&red green:&green blue:&blue alpha:&alpha];
        x.matrix<float>().operator()(0,i) = (255.0 - red) / 255.0f;
        NSLog(@"%f",x.matrix<float>().operator()(0,i));
    }
    // 2. 放入input
    std::vector<std::pair<tensorflow::string, tensorflow::Tensor>> inputs = {
        {"x", x}
    };
    
    std::vector<std::string> nodes = {
        {"softmax"}
    };
    
    const auto start = CACurrentMediaTime();
    
    std::vector<tensorflow::Tensor> outputs;
    // 3. 跑网络
    auto status = session->Run(inputs, nodes, {}, &outputs);
    if (!status.ok()) {
        NSLog(@"Error reading graph: %s", status.error_message().c_str());
        return -1;
    }
    
    NSLog(@"Time: %g seconds", CACurrentMediaTime() - start);
    // 4. 拿到输出，获得结果
    const auto outputMatrix = outputs[0].matrix<float>();
    float bestProbability = 0;
    int bestIndex = -1;
    for (auto i = 0; i < kOutputs; i++) {
        const auto probability = outputMatrix(i);
        if (probability > bestProbability) {
            bestProbability = probability;
            bestIndex = i;
        }
    }
    
    NSLog(@"!!!!!!!!!!! result %d",bestIndex);
    return bestIndex;
}

#pragma mark - load Tensorflow mode

-(BOOL)loadGraphFromPath:(NSString *)path {
    auto status = ReadBinaryProto(tensorflow::Env::Default(), path.fileSystemRepresentation, &graph);
    if (!status.ok()) {
        NSLog(@"Error reading graph: %s", status.error_message().c_str());
        return NO;
    }
    auto nodeCount = graph.node_size();
    NSLog(@"Node count: %d", nodeCount);
    for (auto i = 0; i < nodeCount; ++i) {
        auto node = graph.node(i);
        NSLog(@"Node %d: %s '%s'", i, node.op().c_str(), node.name().c_str());
    }
    return YES;
}

-(BOOL)createSession {
    tensorflow::SessionOptions options;
    auto status = tensorflow::NewSession(options, &session);
    if (!status.ok()) {
        NSLog(@"Error creating session: %s", status.error_message().c_str());
        return NO;
    }
    status = session->Create(graph);
    if (!status.ok()) {
        NSLog(@"Error creating session: %s", status.error_message().c_str());
        return NO;
    }
    return YES;
}

#pragma mark - process image for input

- (NSArray*)getRGBAsFromImage:(UIImage*)image atX:(int)x andY:(int)y count:(int)count
{
    NSMutableArray *result = [NSMutableArray arrayWithCapacity:count];
    
    // First get the image into your data buffer
    CGImageRef imageRef = [image CGImage];
    NSUInteger width = CGImageGetWidth(imageRef);
    NSUInteger height = CGImageGetHeight(imageRef);
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    unsigned char *rawData = (unsigned char*) calloc(height * width * 4, sizeof(unsigned char));
    NSUInteger bytesPerPixel = 4;
    NSUInteger bytesPerRow = bytesPerPixel * width;
    NSUInteger bitsPerComponent = 8;
    CGContextRef context = CGBitmapContextCreate(rawData, width, height,
                                                 bitsPerComponent, bytesPerRow, colorSpace,
                                                 kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
    CGColorSpaceRelease(colorSpace);
    
    CGContextDrawImage(context, CGRectMake(0, 0, width, height), imageRef);
    CGContextRelease(context);
    
    // Now your rawData contains the image data in the RGBA8888 pixel format.
    NSUInteger byteIndex = (bytesPerRow * y) + x * bytesPerPixel;
    for (int i = 0 ; i < count ; ++i)
    {
        //CGFloat alpha = ((CGFloat) rawData[byteIndex + 3] ) / 255.0f;
        CGFloat red   = ((CGFloat) rawData[byteIndex]     ) ;
        CGFloat green = ((CGFloat) rawData[byteIndex + 1] ) ;
        CGFloat blue  = ((CGFloat) rawData[byteIndex + 2] ) ;
        byteIndex += bytesPerPixel;
        
        UIColor *acolor = [UIColor colorWithRed:red green:green blue:blue alpha:1.0];
        [result addObject:acolor];
    }
    
    free(rawData);
    
    return result;
}


- (UIImage *)convertImageToGrayScale:(UIImage *)image {
    // Create image rectangle with current image width/height
    CGRect imageRect = CGRectMake(0, 0, image.size.width, image.size.height);
    
    // Grayscale color space
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceGray();
    
    // Create bitmap content with current image size and grayscale colorspace
    CGContextRef context = CGBitmapContextCreate(nil, image.size.width, image.size.height, 8, 0, colorSpace, kCGImageAlphaNone);
    
    // Draw image into current context, with specified rectangle
    // using previously defined context (with grayscale colorspace)
    CGContextDrawImage(context, imageRect, [image CGImage]);
    
    // Create bitmap image info from pixel data in current context
    CGImageRef imageRef = CGBitmapContextCreateImage(context);
    
    // Create a new UIImage object
    UIImage *newImage = [UIImage imageWithCGImage:imageRef];
    
    // Release colorspace, context and bitmap information
    CGColorSpaceRelease(colorSpace);
    CGContextRelease(context);
    CFRelease(imageRef);
    
    // Return the new grayscale image
    return newImage;
}

- (UIImage *)imageFromColor:(UIColor *)color {
    CGRect rect = CGRectMake(0, 0, 1, 1);
    UIGraphicsBeginImageContext(rect.size);
    CGContextRef context = UIGraphicsGetCurrentContext();
    CGContextSetFillColorWithColor(context, [color CGColor]);
    CGContextFillRect(context, rect);
    UIImage *image = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    return image;
}

-(UIImage *)scaleImage:(UIImage *)image {
    CGSize oldSize = image.size;
    NSLog(@"before scale w %f, y %f",oldSize.width, oldSize.height);
    CGFloat nHeight = 0.0;
    CGFloat nWidth = 0.0;
    if (oldSize.height > oldSize.width) {
        nHeight = 28;
        nWidth = nHeight * (oldSize.width / oldSize.height);
    } else {
        nWidth = 28;
        nHeight = nWidth * (oldSize.height / oldSize.width);
    }
    
    CGRect rect = CGRectMake(0,0,nHeight,nWidth);
    CGRect rectfull = CGRectMake(0, 0, kImageSize, kImageSize);
    UIGraphicsBeginImageContext( rect.size );
    
    UIImage *background = [self imageFromColor:[UIColor whiteColor]];
    [background drawInRect:rectfull];
    [image drawInRect:rect];
    
    UIImage *picture1 = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    
    NSData *imageData = UIImagePNGRepresentation(picture1);
    UIImage *img=[UIImage imageWithData:imageData];
    NSLog(@"after scale w %f, y %f",img.size.width, img.size.height);
    
    return img;
}

@end
