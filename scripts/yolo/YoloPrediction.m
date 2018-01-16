//
//
//  Created by michaelxing on 2018/1/10.
//  Copyright © 2018年 michaelxing. All rights reserved.
//

#import "YoloPrediction.h"
#import "YoloResult.h"
@import CoreML;
@import Vision;

//model 相关
static float anchors[] = {1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52};

static float  confidenceThreshold = 0.7;
static float  iouThreshold = 0.3;

@interface YoloPrediction ()

@property (nonatomic, strong) MLModel *model;
@property (nonatomic, strong) VNCoreMLRequest *MLRequest;
@property (nonatomic, strong) NSMutableArray<YoloResult *> *results;

@end

@implementation YoloPrediction

- (instancetype)initWithModelName:(NSString *)name {
    
    NSArray *paths = [[NSFileManager defaultManager] URLsForDirectory:NSApplicationSupportDirectory inDomains:NSUserDomainMask];
    NSURL *documentsURL = [paths lastObject];
    NSURL *modelcURL = [documentsURL URLByAppendingPathComponent:[NSString stringWithFormat:@"%@.mlmodelc",name]];
    
    return [self initWithModelURL:modelcURL];
}

- (instancetype)initWithModelURL:(NSURL *)url {
    if (self = [super init]) {
        if (url) {
            _model =  [MLModel modelWithContentsOfURL:url error:nil];
            _results = [NSMutableArray arrayWithCapacity:5];
        }
    }
    return self;
}

- (void)predictionObjectRect:(UIImage *)image completionHandler:(void (^)(NSArray<YoloResult *> *, NSError *))completionHandler
{
    VNCoreMLModel *vnModel = [VNCoreMLModel modelForMLModel:_model error:nil];
    
    __weak typeof (self) weakSelf = self;
    self.MLRequest = [[VNCoreMLRequest alloc] initWithModel:vnModel completionHandler:^(VNRequest * _Nonnull request, NSError * _Nullable error) {
        __strong typeof (self) strongSelf = weakSelf;
        
        VNCoreMLFeatureValueObservation *Observation = request.results.firstObject;
        //
        MLMultiArray *featureValues = Observation.featureValue.multiArrayValue;
        NSArray *boundingBoxes = [strongSelf computeBoundingBoxes:featureValues];
        
        dispatch_async(dispatch_get_main_queue(), ^{
            [strongSelf.results removeAllObjects];
            [strongSelf.results addObjectsFromArray:boundingBoxes];
            completionHandler([NSArray arrayWithArray:strongSelf.results],error);
        });
    }];
    
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        __strong typeof (self)strongSelf = weakSelf;
        VNImageRequestHandler *handler = [[VNImageRequestHandler alloc] initWithCIImage:[CIImage imageWithCGImage:image.CGImage]
                                                                            orientation:(CGImagePropertyOrientation)[image imageOrientation]
                                                                                options:@{}];
        [handler performRequests:@[strongSelf.MLRequest] error:nil];
    });
    
    return ;
}

- (NSArray<YoloResult *> *)computeBoundingBoxes:(MLMultiArray *)featureValue {
//    NSAssert(featureValue.count == 125 * 13 * 13, @"feature count error");
    
    float blockSize = 32;
    int gridHeight = 13;
    int gridWidth = 13;
    int boxesPerCell = 5;
    int numClasses = 7;
    double * dataPointer = featureValue.dataPointer;
    
    NSMutableArray *results = [NSMutableArray arrayWithCapacity:5];
    
    for (int cy = 0; cy < gridHeight; cy++)
    {
        for (int cx = 0; cx < gridWidth; cx++)
        {
            for (int b = 0; b < boxesPerCell; b++)
            {
                int  channel = b * (numClasses + 5);
                float tx = dataPointer[[self offset:channel     xStride:cx yStride:cy feature:featureValue]];
                float ty = dataPointer[[self offset:channel + 1 xStride:cx yStride:cy feature:featureValue]];
                float tw = dataPointer[[self offset:channel + 2 xStride:cx yStride:cy feature:featureValue]];
                float th = dataPointer[[self offset:channel + 3 xStride:cx yStride:cy feature:featureValue]];
                float tc = dataPointer[[self offset:channel + 4 xStride:cx yStride:cy feature:featureValue]];
                
                float x = (cx + sigmoid(tx)) * blockSize;
                float y = (cy + sigmoid(ty)) * blockSize;
                
                float w = exp(tw) * anchors[2*b    ] * blockSize;
                float h = exp(th) * anchors[2*b + 1] * blockSize;
                
                float confidence = sigmoid(tc);
                
                float classes[numClasses];
                
                for (int c = 0; c < numClasses; c++)
                {
                    int off = [self offset:channel+5+c xStride:cx yStride:cy feature:featureValue];
                    classes[c] = (float)dataPointer[off];
                }
                softmax(classes, numClasses);
                
                int detectedClass = -1;
                float maxClass = 0;
                for (int c = 0; c < numClasses; ++c) {
                    if (classes[c] > maxClass) {
                        detectedClass = c;
                        maxClass = classes[c];
                    }
                }
                
                float confidenceInClass = maxClass * confidence;
                if (confidenceInClass > confidenceThreshold) {
                    CGRect rect = CGRectMake(x - w/2,
                                             y - h/2,
                                             w,
                                             h);
                    YoloResult *prediction = [YoloResult new];
                    prediction.identifier = detectedClass;
                    prediction.confidence = confidenceInClass;
                    prediction.rect = rect;
                    
                    [results addObject:prediction];
                }
            }
        }
    }
    
    return [self nonMaxSuppression:results limits:3 threhold:0.5];
}

- (NSArray <YoloResult *> *)nonMaxSuppression:(NSArray *)predicts limits:(int)limits threhold:(float)threhold{

    NSArray *sortedPredicts = [predicts sortedArrayUsingComparator:^NSComparisonResult(id  _Nonnull obj1, id  _Nonnull obj2) {
        
        YoloResult *predict_1 = obj1;
        YoloResult *predict_2 = obj2;
        
        return predict_1.confidence > predict_2.confidence;
    }];
    
    NSMutableArray *selectedArray = [NSMutableArray arrayWithCapacity:5];

    bool active[sortedPredicts.count];
    for (int i = 0; i < sortedPredicts.count; i++)
    {
        active[i] = true;
    }
    
    NSInteger numActive = sortedPredicts.count;
    
    for (int i = 0; i < sortedPredicts.count; i++)
    {
        if (active[i]) {
            YoloResult *resultA = [predicts objectAtIndex:i];
            [selectedArray addObject:resultA];
            if(selectedArray.count >= limits) {
                break;
            }
            
            for (int j = i+1; j < sortedPredicts.count; j++)
            {
                if(active[j]) {
                    YoloResult *resultB = [predicts objectAtIndex:j];
                    if ([self IOU:resultA.rect rectB:resultB.rect] > threhold) {
                        active[j] = false;
                        numActive = numActive - 1;
                        if(numActive <= 0) {
                            goto over;
                        }
                    }
                }
            }
        }
    }
    
    over: {
    
    }
    
    return selectedArray;
}

- (int)offset:(int)channel xStride:(int)x yStride:(int)y feature:(MLMultiArray *)feature{
    int  channelStride = feature.strides[0].intValue;
    int yStride = feature.strides[1].intValue;
    int xStride = feature.strides[2].intValue;
    
    return channel * channelStride + y * yStride + x * xStride;
}

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

void softmax(float vals[], int count) {
    float max = -FLT_MAX;
    for (int i=0; i<count; i++) {
        max = fmax(max, vals[i]);
    }
    float sum = 0.0;
    for (int i=0; i<count; i++) {
        vals[i] = exp(vals[i] - max);
        sum += vals[i];
    }
    for (int i=0; i<count; i++) {
        vals[i] /= sum;
    }
}

-(float)IOU:(CGRect)rectA rectB:(CGRect)rectB {
    float areaA = rectA.size.width * rectA.size.height;
    if(areaA <= 0) {
        return 0;
    }
    
    float areaB = rectB.size.width * rectB.size.height;
    if (areaB <= 0) {
        return 0;
    }
    
    float intersectionMinX = MAX(rectA.origin.x, rectB.origin.x);
    float intersectionMinY = MAX(rectA.origin.y, rectB.origin.y);
    float intersectionMaxX = MIN(rectA.origin.x + rectA.size.width, rectB.origin.x + rectB.size.width);
    float intersectionMaxY = MIN(rectA.origin.y + rectA.size.height, rectB.origin.y + rectB.size.height);
    
    float intersectionArea = MAX(intersectionMaxY - intersectionMinY, 0) * MAX(intersectionMaxX - intersectionMinX, 0);
    
    return intersectionArea / (areaA + areaB - intersectionArea);
}

@end
