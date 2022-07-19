#import <UIKit/UIKit.h>

@interface CommonLibraryFactory : NSObject

+ (instancetype)sharedInstance;

- (UIViewController *)getViewControllerInstance;

@end
