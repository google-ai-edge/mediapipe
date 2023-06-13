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

#import <CoreVideo/CoreVideo.h>
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
 * Every mask is has an underlying type which can be accessed using `dataType`. You can access the
 * mask as any other type using the appropriate properties. For eg:, if the underlying type is
 * `MPPMaskDataTypeUInt8`, in addition to accessing the mask using `uint8Array`, you can access
 * 'floatArray` to get the float 32 data. The first time you access the data as a type different
 * from the underlying type, an expensive type conversion is performed. Subsequent accesses return a
 * pointer to the memory location fo the same type converted array. As type conversions can be
 * expensive, it is recommended to limit the accesses to data of types different from the underlying
 * type.
 *
 * Masks that are returned from a MediaPipe Tasks are owned by by the underlying C++ Task. If you
 * need to extend the lifetime of these objects, you can invoke the `[MPPMask copy:]` method.
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
 * stored.
 */
@property(nonatomic, readonly, assign) const UInt8 *uint8Data;

/**
 * The pointer to the memory location where the underlying mask as a single channel float 32 array
 * is stored.
 */
@property(nonatomic, readonly, assign) const float *float32Data;

/**
 * Initializes an `MPPMask` object of tyep `MPPMaskDataTypeUInt8` with the given `UInt8*` data,
 * width and height.
 *
 * @param uint8Data A pointer to the memory location of the `UInt8` data array.
 * @param width The width of the mask.
 * @param height The height of the mask.
 *
 * @return A new `MPPMask` instance with the given `UInt8*` data, width and height.
 */
- (nullable instancetype)initWithUInt8Data:(const UInt8 *)uint8Data
                                     width:(NSInteger)width
                                    height:(NSInteger)height NS_DESIGNATED_INITIALIZER;

/**
 * Initializes an `MPPMask` object of tyep `MPPMaskDataTypeFloat32` with the given `float*` data,
 * width and height.
 *
 * @param uint8Data A pointer to the memory location of the `float` data array.
 * @param width The width of the mask.
 * @param height The height of the mask.
 *
 * @return A new `MPPMask` instance with the given `float*` data, width and height.
 */
- (nullable instancetype)initWithFloat32Data:(const float *)float32Data
                                       width:(NSInteger)width
                                      height:(NSInteger)height
                                       error:(NSError **)error NS_DESIGNATED_INITIALIZER;


// TODO: Add methods for CVPixelBuffer conversion.


/** Unavailable. */
- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
