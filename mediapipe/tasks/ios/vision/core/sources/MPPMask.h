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

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/** The underlying type of the segmentation mask. */
typedef NS_ENUM(NSUInteger, MPPMaskDataType) {

  /** Represents the native `UInt8 *` type. */
  MPPMaskDataTypeUInt8,

  /** Represents the native `float *` type. */
  MPPMaskDataTypeFloat32,

} NS_SWIFT_NAME(MaskDataType);

/**
 * The wrapper class for MediaPipe segmentation masks.
 *
 * Masks are stored as `UInt8 *` or `float *` objects.
 * Every mask has an underlying type which can be accessed using `dataType`. You can access the
 * mask as any other type using the appropriate properties. For example, if the underlying type is
 * `uInt8`, in addition to accessing the mask using `uint8Data`, you can access `float32Data` to get
 * the 32 bit float data (with values ranging from 0.0 to 1.0). The first time you access the data
 * as a type different from the underlying type, an expensive type conversion is performed.
 * Subsequent accesses return a pointer to the memory location for the same type converted array. As
 * type conversions can be expensive, it is recommended to limit the accesses to data of types
 * different from the underlying type.
 *
 * Masks that are returned from a MediaPipe Tasks are owned by by the underlying C++ Task. If you
 * need to extend the lifetime of these objects, you can invoke the `copy()` method.
 */
NS_SWIFT_NAME(Mask)
@interface MPPMask : NSObject <NSCopying>

/** The width of the mask. */
@property(nonatomic, readonly) NSInteger width;

/** The height of the mask. */
@property(nonatomic, readonly) NSInteger height;

/** The data type of the mask. */
@property(nonatomic, readonly) MPPMaskDataType dataType;

/**
 * The pointer to the memory location where the underlying mask as a single channel `UInt8` array is
 * stored. Uint8 values use the full value range and range from 0 to 255.
 */
@property(nonatomic, readonly, assign) const UInt8 *uint8Data;

/**
 * The pointer to the memory location where the underlying mask as a single channel float 32 array
 * is stored. Float values range from 0.0 to 1.0.
 */
@property(nonatomic, readonly, assign) const float *float32Data;

/**
 * Initializes an `Mask` object of type `uInt8` with the given `UInt8*` data, width and height.
 *
 * If `shouldCopy` is set to `true`, the newly created `Mask` stores a reference to a deep copied
 * `uint8Data`. Since deep copies are expensive, it is recommended to not set `shouldCopy` unless
 * the `Mask` must outlive the passed in `uint8Data`.
 *
 * @param uint8Data A pointer to the memory location of the `UInt8` data array.
 * @param width The width of the mask.
 * @param height The height of the mask.
 * @param shouldCopy The height of the mask.
 *
 * @return A new `Mask` instance with the given `UInt8*` data, width and height.
 */
- (nullable instancetype)initWithUInt8Data:(const UInt8 *)uint8Data
                                     width:(NSInteger)width
                                    height:(NSInteger)height
                                shouldCopy:(BOOL)shouldCopy NS_DESIGNATED_INITIALIZER;

/**
 * Initializes an `Mask` object of type `float32` with the given `float*` data, width and height.
 *
 * If `shouldCopy` is set to `true`, the newly created `Mask` stores a reference to a deep copied
 * `float32Data`. Since deep copies are expensive, it is recommended to not set `shouldCopy` unless
 * the `Mask` must outlive the passed in `float32Data`.
 *
 * @param float32Data A pointer to the memory location of the `float` data array.
 * @param width The width of the mask.
 * @param height The height of the mask.
 *
 * @return A new `Mask` instance with the given `float*` data, width and height.
 */
- (nullable instancetype)initWithFloat32Data:(const float *)float32Data
                                       width:(NSInteger)width
                                      height:(NSInteger)height
                                  shouldCopy:(BOOL)shouldCopy NS_DESIGNATED_INITIALIZER;

// TODO: Add methods for CVPixelBuffer conversion.

/** Unavailable. */
- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
