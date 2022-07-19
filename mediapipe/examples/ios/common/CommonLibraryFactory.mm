#import "CommonLibraryFactory.h"
#import "CommonViewController.h"

@implementation CommonLibraryFactory

+ (instancetype)sharedInstance {
  static CommonLibraryFactory* instance = nil;
  static dispatch_once_t onceToken;
  dispatch_once(&onceToken, ^{
    instance = [[CommonLibraryFactory alloc] init];
  });
  return instance;
}

- (UIViewController *)getViewControllerInstance {
    return [CommonViewController new];
}

@end