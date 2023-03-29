// Copyright 2023 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import MPPCommon
import MPPImageTestUtils
import XCTest

@testable import MPPImageClassifier

typealias FileInfo = (name: String, type: String)

class ImageClassifierTests: XCTestCase {

  static let bundle = Bundle(for: ImageClassifierTests.self)

  static let floatModelPath = bundle.path(
    forResource: "mobilenet_v2_1.0_224",
    ofType: "tflite")

  static let quantizedModelPath = bundle.path(
    forResource: "mobilenet_v1_0.25_224_quant",
    ofType: "tflite")

  static let burgerImage = FileInfo(name: "burger", type: "jpg")
  static let burgerRotatedImage = FileInfo(name: "burger_rotated", type: "jpg")
  static let multiObjectsImage = FileInfo(name: "multi_objects", type: "jpg")
  static let multiObjectsRotatedImage = FileInfo(name: "multi_objects_rotated", type: "jpg")

  static let mobileNetCategoriesCount: Int = 1001;

  static let expectedResultsClassifyBurgerImageWithFloatModel = [
    ResultCategory(
      index: 934,
      score: 0.786005,
      categoryName: "cheeseburger",
      displayName: nil),
    ResultCategory(
      index: 932,
      score: 0.023508,
      categoryName: "bagel",
      displayName: nil),
    ResultCategory(
      index: 925,
      score: 0.021172,
      categoryName: "guacamole",
      displayName: nil),
  ]

  func assertEqualErrorDescriptions(
    _ error: Error, expectedLocalizedDescription: String
  ) {
    XCTAssertEqual(
      error.localizedDescription,
      expectedLocalizedDescription)
  }

  func assertCategoriesAreEqual(
    category: ResultCategory,
    expectedCategory: ResultCategory,
    indexInCategoryList: Int
  ) {
    XCTAssertEqual(
      category.index,
      expectedCategory.index,
      String(
        format: """
          category[%d].index and expectedCategory[%d].index are not equal.
          """, indexInCategoryList))
    XCTAssertEqual(
      category.score,
      expectedCategory.score,
      accuracy: 1e-3,
      String(
        format: """
          category[%d].score and expectedCategory[%d].score are not equal.
          """, indexInCategoryList))
    XCTAssertEqual(
      category.categoryName,
      expectedCategory.categoryName,
      String(
        format: """
          category[%d].categoryName and expectedCategory[%d].categoryName are \
          not equal.
          """, indexInCategoryList))
    XCTAssertEqual(
      category.displayName,
      expectedCategory.displayName,
      String(
        format: """
          category[%d].displayName and expectedCategory[%d].displayName are \
          not equal.
          """, indexInCategoryList))
  }

  func assertEqualCategoryArrays(
    categoryArray: [ResultCategory],
    expectedCategoryArray: [ResultCategory]
  ) {
    XCTAssertEqual(
      categoryArray.count,
      expectedCategoryArray.count)

    for (index, (category, expectedCategory)) in 
      zip(categoryArray, expectedCategoryArray)
      .enumerated()
    {
      assertCategoriesAreEqual(
        category: category,
        expectedCategory: expectedCategory,
        indexInCategoryList: index)
    }
  }

  func assertImageClassifierResultHasOneHead(
    _ imageClassifierResult: ImageClassifierResult
  ) {
    XCTAssertEqual(
      imageClassifierResult.classificationResult.classifications.count, 
      1)
    XCTAssertEqual(
      imageClassifierResult.classificationResult.classifications[0].headIndex, 
      0)
  }

  func imageWithFileInfo(_ fileInfo: FileInfo) throws -> MPImage {
    let mpImage = try XCTUnwrap(
      MPImage.imageFromBundle(
        withClass: type(of: self),
        filename: fileInfo.name,
        type: fileInfo.type))

   return mpImage
  }

  func imageClassifierOptionsWithModelPath(
    _ modelPath: String?
  ) throws -> ImageClassifierOptions {
    let modelPath = try XCTUnwrap(modelPath)

    let imageClassifierOptions = ImageClassifierOptions()
    imageClassifierOptions.baseOptions.modelAssetPath = modelPath

    return imageClassifierOptions
  }

  func assertCreateImageClassifierThrowsError(
    imageClassifierOptions: ImageClassifierOptions,
    expectedErrorDescription: String
  ) {
    do {
      let imageClassifier = try ImageClassifier(options: imageClassifierOptions)
      XCTAssertNil(imageClassifier)
    } catch {
      assertEqualErrorDescriptions(
        error,
        expectedLocalizedDescription: expectedErrorDescription)
    }
  }

  func assertImageClassifierResult(
    _ imageClassifierResult: ImageClassifierResult,
    hasCategoryCount expectedCategoryCount: Int,
    andCategories expectedCategories: [ResultCategory]
  ) throws {
    assertImageClassifierResultHasOneHead(imageClassifierResult)
    let categories = 
      imageClassifierResult.classificationResult.classifications[0].categories

    XCTAssertEqual(categories.count, expectedCategoryCount)    
    assertEqualCategoryArrays(
      categoryArray:
        Array(categories.prefix(expectedCategories.count)),
      expectedCategoryArray: expectedCategories)
  }

  func assertResultsForClassifyImage(
    _ image: MPImage,
    usingImageClassifier imageClassifier: ImageClassifier,
    hasCategoryCount expectedCategoryCount: Int,
    andCategories expectedCategories: [ResultCategory]
  ) throws {
    let imageClassifierResult =
      try XCTUnwrap(
        imageClassifier.classify(image: image))
    
    try assertImageClassifierResult(
      imageClassifierResult,
      hasCategoryCount: expectedCategoryCount,
      andCategories: expectedCategories
    )
  }

  func assertResultsForClassifyImageWithFileInfo(
    _ fileInfo: FileInfo,
    usingImageClassifier imageClassifier: ImageClassifier,
    hasCategoryCount expectedCategoryCount: Int,
    andCategories expectedCategories: [ResultCategory]
  ) throws {
    let mpImage = try XCTUnwrap(
      imageWithFileInfo(fileInfo))

    try assertResultsForClassifyImage(
      mpImage,
      usingImageClassifier: imageClassifier,
      hasCategoryCount: expectedCategoryCount,
      andCategories: expectedCategories
    )
  }

  func testCreateImageClassifierWithCategoryAllowlistAndDenylistFails() throws {

    let imageClassifierOptions =
      try XCTUnwrap(
        imageClassifierOptionsWithModelPath(ImageClassifierTests.floatModelPath))
    imageClassifierOptions.categoryAllowlist = ["bagel"]
    imageClassifierOptions.categoryDenylist = ["guacamole"]

    assertCreateImageClassifierThrowsError(
      imageClassifierOptions: imageClassifierOptions,
      expectedErrorDescription: """
        INVALID_ARGUMENT: `category_allowlist` and `category_denylist` are \
        mutually exclusive options.
        """)
  }

  func testClassifyWithModelPathAndFloatModelSucceeds() throws {

    let modelPath = try XCTUnwrap(ImageClassifierTests.floatModelPath)
    let imageClassifier = try XCTUnwrap(ImageClassifier(modelPath: modelPath))

    try assertResultsForClassifyImageWithFileInfo(
      ImageClassifierTests.burgerImage,
      usingImageClassifier: imageClassifier,
      hasCategoryCount: ImageClassifierTests.mobileNetCategoriesCount,
      andCategories: ImageClassifierTests.expectedResultsClassifyBurgerImageWithFloatModel)
  }

  func testClassifyWithOptionsAndFloatModelSucceeds() throws {

    let imageClassifierOptions =
      try XCTUnwrap(
        imageClassifierOptionsWithModelPath(ImageClassifierTests.floatModelPath))

    let imageClassifier = try XCTUnwrap(ImageClassifier(options: imageClassifierOptions))

    try assertResultsForClassifyImageWithFileInfo(
      ImageClassifierTests.burgerImage,
      usingImageClassifier: imageClassifier,
      hasCategoryCount: ImageClassifierTests.mobileNetCategoriesCount,
      andCategories: 
        ImageClassifierTests.expectedResultsClassifyBurgerImageWithFloatModel)
  }

  func testClassifyWithQuantizedModelSucceeds() throws {

    let imageClassifierOptions =
      try XCTUnwrap(
        imageClassifierOptionsWithModelPath(
          ImageClassifierTests.quantizedModelPath))

    let imageClassifier = try XCTUnwrap(ImageClassifier(options: imageClassifierOptions))

    let expectedCategories = [
      ResultCategory(
        index: 934,
        score: 0.972656,
        categoryName: "cheeseburger",
        displayName: nil),
    ]

    try assertResultsForClassifyImageWithFileInfo(
      ImageClassifierTests.burgerImage,
      usingImageClassifier: imageClassifier,
      hasCategoryCount: ImageClassifierTests.mobileNetCategoriesCount,
      andCategories: expectedCategories)
  }

  func testClassifyWithScoreThresholdSucceeds() throws {

    let imageClassifierOptions =
      try XCTUnwrap(
        imageClassifierOptionsWithModelPath(ImageClassifierTests.floatModelPath))
    imageClassifierOptions.scoreThreshold = 0.25

    let imageClassifier = try XCTUnwrap(ImageClassifier(
      options: imageClassifierOptions))

    let expectedCategories = [
      ResultCategory(
          index: 934,
          score: 0.786005,
          categoryName: "cheeseburger",
          displayName: nil),
      ]

    try assertResultsForClassifyImageWithFileInfo(
      ImageClassifierTests.burgerImage,
      usingImageClassifier: imageClassifier,
      hasCategoryCount: expectedCategories.count,
      andCategories: expectedCategories)
  }

  func testClassifyWithAllowlistSucceeds() throws {

    let imageClassifierOptions =
      try XCTUnwrap(
        imageClassifierOptionsWithModelPath(ImageClassifierTests.floatModelPath))
    imageClassifierOptions.categoryAllowlist = 
      ["cheeseburger", "guacamole", "meat loaf"]

    let imageClassifier = try XCTUnwrap(ImageClassifier(
      options: imageClassifierOptions))

    let expectedCategories = [
      ResultCategory(
          index: 934,
          score: 0.786005,
          categoryName: "cheeseburger",
          displayName: nil),
      ResultCategory(
          index: 925,
          score: 0.021172,
          categoryName: "guacamole",
          displayName: nil),
      ResultCategory(
          index: 963,
          score: 0.006279315,
          categoryName: "meat loaf",
          displayName: nil),
      ]

    try assertResultsForClassifyImageWithFileInfo(
      ImageClassifierTests.burgerImage,
      usingImageClassifier: imageClassifier,
      hasCategoryCount: expectedCategories.count,
      andCategories: expectedCategories)
  }

  func testClassifyWithDenylistSucceeds() throws {

    let imageClassifierOptions =
      try XCTUnwrap(
        imageClassifierOptionsWithModelPath(
          ImageClassifierTests.floatModelPath))
    imageClassifierOptions.categoryDenylist = ["bagel"]

    let maxResults = 3;
    imageClassifierOptions.maxResults = maxResults;

    let imageClassifier = try XCTUnwrap(ImageClassifier(
      options: imageClassifierOptions))

    let expectedCategories = [
      ResultCategory(
          index: 934,
          score: 0.786005,
          categoryName: "cheeseburger",
          displayName: nil),
      ResultCategory(
          index: 925,
          score: 0.021172,
          categoryName: "guacamole",
          displayName: nil),
      ResultCategory(
          index: 963,
          score: 0.006279315,
          categoryName: "meat loaf",
          displayName: nil),
      ]

    try assertResultsForClassifyImageWithFileInfo(
      ImageClassifierTests.burgerImage,
      usingImageClassifier: imageClassifier,
      hasCategoryCount: maxResults,
      andCategories: expectedCategories)
  }

  func testClassifyWithRegionOfInterestSucceeds() throws {

    let imageClassifierOptions =
      try XCTUnwrap(
        imageClassifierOptionsWithModelPath(
          ImageClassifierTests.floatModelPath))

    let maxResults = 1;
    imageClassifierOptions.maxResults = maxResults;

    let imageClassifier = try XCTUnwrap(ImageClassifier(
      options: imageClassifierOptions))

    let mpImage = try XCTUnwrap(
      imageWithFileInfo(ImageClassifierTests.multiObjectsImage))

    let imageClassifierResult =  try XCTUnwrap(
        imageClassifier.classify(
          image: mpImage,
          regionOfInterest: CGRect(
            x: 0.450, 
            y: 0.308, 
            width: 0.164, 
            height: 0.426)))

    let expectedCategories = [
      ResultCategory(
          index: 806,
          score: 0.997122,
          categoryName: "soccer ball",
          displayName: nil),
      ]


    try assertImageClassifierResult(
      imageClassifierResult,
      hasCategoryCount: maxResults,
      andCategories: expectedCategories)
  }

  func testClassifyWithOrientationSucceeds() throws {

    let imageClassifierOptions =
      try XCTUnwrap(
        imageClassifierOptionsWithModelPath(
          ImageClassifierTests.floatModelPath))

    let maxResults = 3;
    imageClassifierOptions.maxResults = maxResults;

    let imageClassifier = try XCTUnwrap(ImageClassifier(
      options: imageClassifierOptions))

    let expectedCategories = [
      ResultCategory(
          index: 934,
          score: 0.622074,
          categoryName: "cheeseburger",
          displayName: nil),
      ResultCategory(
          index: 963,
          score: 0.051214,
          categoryName: "meat loaf",
          displayName: nil),
      ResultCategory(
          index: 925,
          score: 0.048719,
          categoryName: "guacamole",
          displayName: nil),
      ]

    let mpImage = try XCTUnwrap(
      MPImage.imageFromBundle(
        withClass: type(of: self),
        filename: ImageClassifierTests.burgerRotatedImage.name,
        type: ImageClassifierTests.burgerRotatedImage.type,
        orientation: .right))

     try assertResultsForClassifyImage(
      mpImage,
      usingImageClassifier: imageClassifier,
      hasCategoryCount: expectedCategories.count,
      andCategories: expectedCategories)
  }

  func testClassifyWithOrientationAndRegionOfInterestSucceeds() throws {

    let imageClassifierOptions =
      try XCTUnwrap(
        imageClassifierOptionsWithModelPath(
          ImageClassifierTests.floatModelPath))

    let maxResults = 3;
    imageClassifierOptions.maxResults = maxResults;

    let imageClassifier = try XCTUnwrap(ImageClassifier(
      options: imageClassifierOptions))

    let expectedCategories = [
      ResultCategory(
          index: 560,
          score: 0.682305,
          categoryName: "folding chair",
          displayName: nil),
      ]

    let mpImage = try XCTUnwrap(
      MPImage.imageFromBundle(
        withClass: type(of: self),
        filename: ImageClassifierTests.multiObjectsRotatedImage.name,
        type: ImageClassifierTests.multiObjectsRotatedImage.type,
        orientation: .right))

     let imageClassifierResult =  try XCTUnwrap(
        imageClassifier.classify(
          image: mpImage,
          regionOfInterest: CGRect(
            x: 0.0, 
            y: 0.1763, 
            width: 0.5642, 
            height: 0.1286)))


    try assertImageClassifierResult(
      imageClassifierResult,
      hasCategoryCount: maxResults,
      andCategories: expectedCategories)
  }

  func testImageClassifierFailsWithResultListenerInNonLiveStreamMode() throws {

    let runningModesToTest = [RunningMode.image, RunningMode.video];

    for runningMode in runningModesToTest {
      let imageClassifierOptions =
      try XCTUnwrap(
        imageClassifierOptionsWithModelPath(
          ImageClassifierTests.floatModelPath))
      imageClassifierOptions.runningMode = runningMode
      imageClassifierOptions.completion = {(
        result: ImageClassifierResult?, 
        timestampMs: Int,
        error: Error?) -> () in
      } 

      assertCreateImageClassifierThrowsError(
      imageClassifierOptions: imageClassifierOptions,
      expectedErrorDescription: """
        The vision task is in image or video mode, a user-defined result \
        callback should not be provided.
        """)
    }
  }

  func testImageClassifierFailsWithMissingResultListenerInLiveStreamMode() 
  throws {

    let imageClassifierOptions =
    try XCTUnwrap(
      imageClassifierOptionsWithModelPath(
        ImageClassifierTests.floatModelPath))
    imageClassifierOptions.runningMode = .liveStream

    assertCreateImageClassifierThrowsError(
    imageClassifierOptions: imageClassifierOptions,
    expectedErrorDescription: """
      The vision task is in live stream mode, a user-defined result callback \
      must be provided.
      """)
  }

  func testClassifyFailsWithCallingWrongApiInImageMode() throws {

    let imageClassifierOptions =
    try XCTUnwrap(
      imageClassifierOptionsWithModelPath(
        ImageClassifierTests.floatModelPath))

    let imageClassifier = try XCTUnwrap(ImageClassifier(options: 
      imageClassifierOptions))

    let mpImage = try XCTUnwrap(
      imageWithFileInfo(ImageClassifierTests.multiObjectsImage))

    do {
      try imageClassifier.classifyAsync(
       image: mpImage,
       timestampMs:0)
    } catch {
      assertEqualErrorDescriptions(
        error,
        expectedLocalizedDescription: """
      The vision task is not initialized with live stream mode. Current \
      Running Mode: Image
      """)
    }

    do {
      let imagClassifierResult = try imageClassifier.classify(
        videoFrame: mpImage,
        timestampMs: 0)
      XCTAssertNil(imagClassifierResult)
    } catch {
      assertEqualErrorDescriptions(
        error,
        expectedLocalizedDescription: """
      The vision task is not initialized with video mode. Current Running \
      Mode: Image
      """)
    }
  }

  func testClassifyFailsWithCallingWrongApiInVideoMode() throws {

    let imageClassifierOptions =
    try XCTUnwrap(
      imageClassifierOptionsWithModelPath(
        ImageClassifierTests.floatModelPath))
    
    imageClassifierOptions.runningMode = .video

    let imageClassifier = try XCTUnwrap(ImageClassifier(options: 
      imageClassifierOptions))

    let mpImage = try XCTUnwrap(
      imageWithFileInfo(ImageClassifierTests.multiObjectsImage))

    do {
      try imageClassifier.classifyAsync(
       image: mpImage,
       timestampMs:0)
    } catch {
      assertEqualErrorDescriptions(
        error,
        expectedLocalizedDescription: """
      The vision task is not initialized with live stream mode. Current \
      Running Mode: Video
      """)
    }

    do {
      let imagClassifierResult = try imageClassifier.classify(
        image: mpImage)
      XCTAssertNil(imagClassifierResult)
    } catch {
      assertEqualErrorDescriptions(
        error,
        expectedLocalizedDescription: """
      The vision task is not initialized with image mode. Current Running \
      Mode: Video
      """)
    }
  }

  func testClassifyFailsWithCallingWrongApiLiveStreamInMode() throws {
    let imageClassifierOptions =
    try XCTUnwrap(
      imageClassifierOptionsWithModelPath(
        ImageClassifierTests.floatModelPath))
    
    imageClassifierOptions.runningMode = .liveStream
    imageClassifierOptions.completion = {(
      result: ImageClassifierResult?, 
      timestampMs: Int,
      error: Error?) -> () in
    } 

    let imageClassifier = try XCTUnwrap(ImageClassifier(options: 
      imageClassifierOptions))

    let mpImage = try XCTUnwrap(
      imageWithFileInfo(ImageClassifierTests.multiObjectsImage))

    do {
      let imagClassifierResult = try imageClassifier.classify(
       image: mpImage)
      XCTAssertNil(imagClassifierResult)
    } catch {
      assertEqualErrorDescriptions(
        error,
        expectedLocalizedDescription: """
      The vision task is not initialized with image mode. Current Running \
      Mode: Live Stream
      """)
    }

    do {
      let imagClassifierResult = try imageClassifier.classify(
        videoFrame: mpImage,
        timestampMs: 0)
      XCTAssertNil(imagClassifierResult)
    } catch {
      assertEqualErrorDescriptions(
        error,
        expectedLocalizedDescription: """
      The vision task is not initialized with video mode. Current Running \
      Mode: Live Stream
      """)
    }
  }

  func testClassifyWithVideoModeSucceeds() throws {
    let imageClassifierOptions =
    try XCTUnwrap(
      imageClassifierOptionsWithModelPath(
        ImageClassifierTests.floatModelPath))
    
    imageClassifierOptions.runningMode = .video

    let maxResults = 3;
    imageClassifierOptions.maxResults = maxResults
  
    let imageClassifier = try XCTUnwrap(ImageClassifier(options: 
      imageClassifierOptions))

    let mpImage = try XCTUnwrap(
      imageWithFileInfo(ImageClassifierTests.burgerImage))
    
    for i in 0..<3 {
      let imageClassifierResult = try XCTUnwrap(
        imageClassifier.classify(
          videoFrame: mpImage,
          timestampMs: i))
      try assertImageClassifierResult(
        imageClassifierResult,
        hasCategoryCount: maxResults,
        andCategories: 
          ImageClassifierTests.expectedResultsClassifyBurgerImageWithFloatModel
      )
    }
  }

  func testClassifyWithOutOfOrderTimestampsAndLiveStreamModeSucceeds() throws {
    let imageClassifierOptions =
    try XCTUnwrap(
      imageClassifierOptionsWithModelPath(
        ImageClassifierTests.floatModelPath))
    
    imageClassifierOptions.runningMode = .liveStream

    let maxResults = 3
    imageClassifierOptions.maxResults = maxResults
 
    let expectation = expectation(
      description: "classifyWithOutOfOrderTimestampsAndLiveStream")
    expectation.expectedFulfillmentCount = 1;
  
    imageClassifierOptions.completion = {(
      result: ImageClassifierResult?, 
      timestampMs: Int, 
      error: Error?) -> () in
      do {
        try self.assertImageClassifierResult(
          try XCTUnwrap(result),
          hasCategoryCount: maxResults,
          andCategories: 
            ImageClassifierTests
              .expectedResultsClassifyBurgerImageWithFloatModel)
      }
      catch {
          // Any errors will be thrown by the wait() method of the expectation.
      }
     expectation.fulfill()
    }
  
    let imageClassifier = try XCTUnwrap(ImageClassifier(options: 
      imageClassifierOptions))

    let mpImage = try XCTUnwrap(
      imageWithFileInfo(ImageClassifierTests.burgerImage))

    XCTAssertNoThrow(
      try imageClassifier.classifyAsync(
        image: mpImage,
        timestampMs: 100))
      
    XCTAssertThrowsError(
      try imageClassifier.classifyAsync(
        image: mpImage,
        timestampMs: 0)) {(error) in
          assertEqualErrorDescriptions(
          error,
          expectedLocalizedDescription: """
          INVALID_ARGUMENT: Input timestamp must be monotonically \
          increasing.
          """)
    }
   
    wait(for:[expectation], timeout: 0.1)
  }

  func testClassifyWithLiveStreamModeSucceeds() throws {
    let imageClassifierOptions =
    try XCTUnwrap(
      imageClassifierOptionsWithModelPath(
        ImageClassifierTests.floatModelPath))
    
    imageClassifierOptions.runningMode = .liveStream

    let maxResults = 3
    imageClassifierOptions.maxResults = maxResults

    let iterationCount = 100;
 
   // Because of flow limiting, we cannot ensure that the callback will be 
   // invoked `iterationCount` times.
   // An normal expectation will fail if expectation.fullfill() is not called 
   // `expectation.expectedFulfillmentCount` times.
   // If `expectation.isInverted = true`, the test will only succeed if 
   // expectation is not fullfilled for the specified `expectedFulfillmentCount`.
   // Since in our case we cannot predict how many times the expectation is 
   // supposed to be fullfilled setting,
   // `expectation.expectedFulfillmentCount` = `iterationCount` and 
   // `expectation.isInverted = true` ensures that test succeeds if 
   // expectation is not fullfilled `iterationCount` times.
    let expectation = expectation(description: "liveStreamClassify")
    expectation.expectedFulfillmentCount = iterationCount;
    expectation.isInverted = true;
  
    imageClassifierOptions.completion = {(
      result: ImageClassifierResult?, 
      timestampMs: Int, 
      error: Error?) -> () in
      do {
        try self.assertImageClassifierResult(
          try XCTUnwrap(result),
          hasCategoryCount: maxResults,
          andCategories: 
            ImageClassifierTests
              .expectedResultsClassifyBurgerImageWithFloatModel)
      }
      catch {
          // Any errors will be thrown by the wait() method of the expectation.
      }
     expectation.fulfill()
    }
  
    let imageClassifier = try XCTUnwrap(ImageClassifier(options: 
      imageClassifierOptions))

     let mpImage = try XCTUnwrap(
      imageWithFileInfo(ImageClassifierTests.burgerImage))
    
    for i in 0..<iterationCount {
      XCTAssertNoThrow(
        try imageClassifier.classifyAsync(
          image: mpImage,
          timestampMs: i))
    }

    wait(for:[expectation], timeout: 0.5)
  }
}
