//
//  ViewController.m
//  MediapipeiOSPlayground
//
//  Created by 王韧竹 on 2022/7/15.
//

#import "ViewController.h"
#import <CommonMediaPipeFramework/CommonLibraryFactory.h>

@interface ViewController ()

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view.
}

- (void)viewDidAppear:(BOOL)animated {
    UIViewController *vc = [[CommonLibraryFactory sharedInstance] getViewControllerInstance];
    [self.navigationController pushViewController:vc animated:YES];
}


@end
