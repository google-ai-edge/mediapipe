// Copyright 2019 The MediaPipe Authors.
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

#include "mediapipe/framework/collection.h"

#include "absl/strings/str_cat.h"
#include "mediapipe/framework/packet_set.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/tool/tag_map_helper.h"

namespace mediapipe {
namespace {

constexpr char kTag2Tag[] = "TAG_2";
constexpr char kTag0Tag[] = "TAG_0";
constexpr char kTag1Tag[] = "TAG_1";

TEST(CollectionTest, BasicByIndex) {
  tool::TagAndNameInfo info;
  info.names.push_back("name_1");
  info.names.push_back("name_0");
  info.names.push_back("name_2");
  internal::Collection<int> collection(info);
  collection.Index(1) = 101;
  collection.Index(0) = 100;
  collection.Index(2) = 102;

  // Test the stored values.
  EXPECT_EQ(100, collection.Index(0));
  EXPECT_EQ(101, collection.Index(1));
  EXPECT_EQ(102, collection.Index(2));
  // Test access using a range based for.
  int i = 0;
  for (int num : collection) {
    EXPECT_EQ(100 + i, num);
    ++i;
  }
}

TEST(CollectionTest, BasicByTag) {
  tool::TagAndNameInfo info;
  info.names.push_back("name_1");
  info.tags.push_back("TAG_1");
  info.names.push_back("name_0");
  info.tags.push_back("TAG_0");
  info.names.push_back("name_2");
  info.tags.push_back("TAG_2");
  internal::Collection<int> collection(info);
  collection.Tag(kTag1Tag) = 101;
  collection.Tag(kTag0Tag) = 100;
  collection.Tag(kTag2Tag) = 102;

  // Test the stored values.
  EXPECT_EQ(100, collection.Tag(kTag0Tag));
  EXPECT_EQ(101, collection.Tag(kTag1Tag));
  EXPECT_EQ(102, collection.Tag(kTag2Tag));
  // Test access using a range based for.
  int i = 0;
  for (int num : collection) {
    // Numbers are in sorted order by tag.
    EXPECT_EQ(100 + i, num);
    ++i;
  }
}

TEST(CollectionTest, MixedTagAndIndexUsage) {
  auto tags_statusor =
      tool::CreateTagMap({"TAG_A:a", "TAG_B:1:b", "TAG_A:2:c", "TAG_B:d",
                          "TAG_C:0:e", "TAG_A:1:f"});
  MP_ASSERT_OK(tags_statusor);

  internal::Collection<int> collection1(std::move(tags_statusor.value()));
  collection1.Get("TAG_A", 0) = 100;
  collection1.Get("TAG_A", 1) = 101;
  collection1.Get("TAG_A", 2) = 102;
  collection1.Get("TAG_B", 0) = 103;
  collection1.Get("TAG_B", 1) = 104;
  collection1.Get("TAG_C", 0) = 105;

  // Test access using a range based for.
  int i = 0;
  for (int num : collection1) {
    // Numbers are in sorted order by tag and then index.
    EXPECT_EQ(100 + i, num);
    ++i;
  }
  EXPECT_EQ(6, i);
  // Initialize the values of another collection while iterating through
  // the entries of the first.  This is testing that two collections
  // can be looped through in lock step.
  internal::Collection<char> collection2(collection1.TagMap());
  i = 0;
  for (CollectionItemId id = collection1.BeginId(); id < collection1.EndId();
       ++id) {
    // Numbers are in sorted order by tag and then index.
    EXPECT_EQ(100 + i, collection1.Get(id));
    // Initialize the entries of the second collection.
    collection2.Get(id) = 'a' + i;
    ++i;
  }
  EXPECT_EQ(6, i);

  // Check the second collection.
  EXPECT_EQ(6, collection2.NumEntries());
  EXPECT_EQ('a', collection2.Get("TAG_A", 0));
  EXPECT_EQ('b', collection2.Get("TAG_A", 1));
  EXPECT_EQ('c', collection2.Get("TAG_A", 2));
  EXPECT_EQ('d', collection2.Get("TAG_B", 0));
  EXPECT_EQ('e', collection2.Get("TAG_B", 1));
  EXPECT_EQ('f', collection2.Get("TAG_C", 0));
  // And check it again with a loop.
  i = 0;
  for (int num : collection2) {
    EXPECT_EQ('a' + i, num);
    ++i;
  }
  EXPECT_EQ(6, i);

  // Initialize the values of another collection by iterating over
  // each tag.
  internal::Collection<std::string> collection3(collection1.TagMap());
  i = 0;
  for (const std::string& tag : collection1.GetTags()) {
    int index_in_tag = 0;
    for (CollectionItemId id = collection1.BeginId(tag);
         id < collection1.EndId(tag); ++id) {
      VLOG(1) << "tag: " << tag << " index_in_tag: " << index_in_tag
              << " collection index: " << i;
      // Numbers are in sorted order by tag and then index.
      EXPECT_EQ(100 + i, collection1.Get(id));
      // Initialize the entries of the second collection.
      collection3.Get(id) = absl::StrCat(i, " ", tag, " ", index_in_tag);
      ++i;
      ++index_in_tag;
    }
  }
  EXPECT_EQ(6, i);

  for (CollectionItemId id = collection1.BeginId("TAG_D");
       id < collection1.EndId("TAG_D"); ++id) {
    EXPECT_FALSE(true) << "iteration through non-existent tag found element.";
  }

  // Check the second collection.
  EXPECT_EQ(6, collection3.NumEntries());
  EXPECT_EQ("0 TAG_A 0", collection3.Get("TAG_A", 0));
  EXPECT_EQ("1 TAG_A 1", collection3.Get("TAG_A", 1));
  EXPECT_EQ("2 TAG_A 2", collection3.Get("TAG_A", 2));
  EXPECT_EQ("3 TAG_B 0", collection3.Get("TAG_B", 0));
  EXPECT_EQ("4 TAG_B 1", collection3.Get("TAG_B", 1));
  EXPECT_EQ("5 TAG_C 0", collection3.Get("TAG_C", 0));
}

TEST(CollectionTest, StaticEmptyCollectionHeapCheck) {
  // Ensure that static collections play nicely with the heap checker.
  // "new T[0]" returns a non-null pointer which the heap checker has
  // issues in tracking.  Additionally, allocating of empty arrays is
  // also inefficient as it invokes heap management routines.
  static auto* collection1 = new PacketSet(tool::CreateTagMap({}).value());
  // Heap check issues are most triggered when zero length and non-zero
  // length allocations are interleaved.  Additionally, this heap check
  // wasn't triggered by "char", so a more complex type (Packet) is used.
  static auto* collection2 =
      new PacketSet(tool::CreateTagMap({"TAG:name"}).value());
  static auto* collection3 = new PacketSet(tool::CreateTagMap({}).value());
  static auto* collection4 =
      new PacketSet(tool::CreateTagMap({"TAG:name"}).value());
  static auto* collection5 = new PacketSet(tool::CreateTagMap({}).value());
  EXPECT_EQ(0, collection1->NumEntries());
  EXPECT_EQ(1, collection2->NumEntries());
  EXPECT_EQ(0, collection3->NumEntries());
  EXPECT_EQ(1, collection4->NumEntries());
  EXPECT_EQ(0, collection5->NumEntries());
}

template <typename T>
absl::Status TestCollectionWithPointers(const std::vector<T>& original_values,
                                        const T& inject1, const T& inject2) {
  std::shared_ptr<tool::TagMap> tag_map =
      tool::CreateTagMap({"TAG_A:a", "TAG_B:1:b", "TAG_A:2:c", "TAG_B:d",
                          "TAG_C:0:e", "TAG_A:1:f"})
          .value();

  {
    // Test a regular collection.
    std::vector<T> values = original_values;
    internal::Collection<T> collection(tag_map);
    collection.Get("TAG_A", 0) = values[0];
    collection.Get("TAG_A", 1) = values[1];
    collection.Get("TAG_A", 2) = values[2];
    collection.Get("TAG_B", 0) = values[3];
    collection.Get("TAG_B", 1) = values[4];
    collection.Get("TAG_C", 0) = values[5];

    const auto* collection_ptr = &collection;

    EXPECT_EQ(values[0], collection.Get("TAG_A", 0));
    EXPECT_EQ(values[1], collection.Get("TAG_A", 1));
    EXPECT_EQ(values[2], collection.Get("TAG_A", 2));
    EXPECT_EQ(values[3], collection.Get("TAG_B", 0));
    EXPECT_EQ(values[4], collection.Get("TAG_B", 1));
    EXPECT_EQ(values[5], collection.Get("TAG_C", 0));

    EXPECT_EQ(values[0], collection_ptr->Get("TAG_A", 0));
    EXPECT_EQ(values[1], collection_ptr->Get("TAG_A", 1));
    EXPECT_EQ(values[2], collection_ptr->Get("TAG_A", 2));
    EXPECT_EQ(values[3], collection_ptr->Get("TAG_B", 0));
    EXPECT_EQ(values[4], collection_ptr->Get("TAG_B", 1));
    EXPECT_EQ(values[5], collection_ptr->Get("TAG_C", 0));

    // Test const-ness.
    EXPECT_EQ(false, std::is_const<typename std::remove_reference<
                         decltype(collection.Get("TAG_A", 0))>::type>::value);
    EXPECT_EQ(true,
              std::is_const<typename std::remove_reference<
                  decltype(collection_ptr->Get("TAG_A", 0))>::type>::value);

    // Test access using a range based for.
    int i = 0;
    for (auto& value : *collection_ptr) {
      EXPECT_EQ(values[i], value);
      EXPECT_EQ(
          true,
          std::is_const<
              typename std::remove_reference<decltype(value)>::type>::value);
      ++i;
    }
    i = 0;
    for (auto& value : collection) {
      EXPECT_EQ(values[i], value);
      EXPECT_EQ(
          false,
          std::is_const<
              typename std::remove_reference<decltype(value)>::type>::value);
      ++i;
    }
    // Test the random access operator in the iterator.
    // the operator[] should not generally be used.
    EXPECT_EQ(values[2], collection_ptr->begin()[2]);
    collection.begin()[2] = inject2;
    EXPECT_EQ(inject2, collection_ptr->Get("TAG_A", 2));
  }

  {
    // Pointer Collection type with dereference_content set to true.
    std::vector<T> values = original_values;
    internal::Collection<T, internal::CollectionStorage::kStorePointer>
        collection(tag_map);
    collection.GetPtr(collection.GetId("TAG_A", 0)) = &values[0];
    collection.GetPtr(collection.GetId("TAG_A", 1)) = &values[1];
    collection.GetPtr(collection.GetId("TAG_A", 2)) = &values[2];
    collection.GetPtr(collection.GetId("TAG_B", 0)) = &values[3];
    collection.GetPtr(collection.GetId("TAG_B", 1)) = &values[4];
    collection.GetPtr(collection.GetId("TAG_C", 0)) = &values[5];

    const auto* collection_ptr = &collection;

    EXPECT_EQ(values[0], collection.Get("TAG_A", 0));
    EXPECT_EQ(values[1], collection.Get("TAG_A", 1));
    EXPECT_EQ(values[2], collection.Get("TAG_A", 2));
    EXPECT_EQ(values[3], collection.Get("TAG_B", 0));
    EXPECT_EQ(values[4], collection.Get("TAG_B", 1));
    EXPECT_EQ(values[5], collection.Get("TAG_C", 0));

    EXPECT_EQ(values[0], collection_ptr->Get("TAG_A", 0));
    EXPECT_EQ(values[1], collection_ptr->Get("TAG_A", 1));
    EXPECT_EQ(values[2], collection_ptr->Get("TAG_A", 2));
    EXPECT_EQ(values[3], collection_ptr->Get("TAG_B", 0));
    EXPECT_EQ(values[4], collection_ptr->Get("TAG_B", 1));
    EXPECT_EQ(values[5], collection_ptr->Get("TAG_C", 0));

    // Test const-ness.
    EXPECT_EQ(false, std::is_const<typename std::remove_reference<
                         decltype(collection.Get("TAG_A", 0))>::type>::value);
    EXPECT_EQ(true,
              std::is_const<typename std::remove_reference<
                  decltype(collection_ptr->Get("TAG_A", 0))>::type>::value);

    // Test access using a range based for.
    int i = 0;
    for (auto& value : *collection_ptr) {
      EXPECT_EQ(values[i], value);
      EXPECT_EQ(
          true,
          std::is_const<
              typename std::remove_reference<decltype(value)>::type>::value);
      ++i;
    }
    i = 0;
    for (auto& value : collection) {
      EXPECT_EQ(values[i], value);
      EXPECT_EQ(
          false,
          std::is_const<
              typename std::remove_reference<decltype(value)>::type>::value);
      ++i;
    }
    i = 0;
    for (CollectionItemId id = collection_ptr->BeginId();
         id < collection_ptr->EndId(); ++id) {
      // TODO Test that GetPtr() does not exist for
      // storage == kStoreValue.
      EXPECT_EQ(&values[i], collection_ptr->GetPtr(id));
      EXPECT_EQ(values[i], *collection_ptr->GetPtr(id));
      EXPECT_EQ(false, std::is_const<typename std::remove_reference<
                           decltype(*collection.GetPtr(id))>::type>::value);
      EXPECT_EQ(true, std::is_const<typename std::remove_reference<
                          decltype(*collection_ptr->GetPtr(id))>::type>::value);
      ++i;
    }

    T injected = inject1;
    collection.GetPtr(collection_ptr->GetId("TAG_A", 2)) = &injected;
    EXPECT_EQ(&injected,
              collection_ptr->GetPtr(collection_ptr->GetId("TAG_A", 2)));
    EXPECT_EQ(injected,
              *collection_ptr->GetPtr(collection_ptr->GetId("TAG_A", 2)));
    EXPECT_EQ(injected, collection_ptr->Get("TAG_A", 2));
    // Test the random access operator in the iterator.
    // the operator[] should not generally be used.
    EXPECT_EQ(
        injected,
        collection_ptr->begin()[collection_ptr->GetId("TAG_A", 2).value()]);
    collection.begin()[collection_ptr->GetId("TAG_A", 2).value()] = inject2;
    EXPECT_EQ(inject2, injected);

    // Test access using a range based for.
    i = 0;
    for (const T& value : *collection_ptr) {
      if (i != collection_ptr->GetId("TAG_A", 2).value()) {
        EXPECT_EQ(values[i], value);
      } else {
        EXPECT_EQ(injected, value);
      }
      ++i;
    }
  }

  {
    // Pointer Collection type with dereference_content set to false.
    std::vector<T> values = original_values;
    internal::Collection<T*, internal::CollectionStorage::kStoreValue>
        collection(tag_map);
    collection.Get("TAG_A", 0) = &values[0];
    collection.Get("TAG_A", 1) = &values[1];
    collection.Get("TAG_A", 2) = &values[2];
    collection.Get("TAG_B", 0) = &values[3];
    collection.Get("TAG_B", 1) = &values[4];
    collection.Get("TAG_C", 0) = &values[5];

    const auto* collection_ptr = &collection;

    EXPECT_EQ(values[0], *collection.Get("TAG_A", 0));
    EXPECT_EQ(values[1], *collection.Get("TAG_A", 1));
    EXPECT_EQ(values[2], *collection.Get("TAG_A", 2));
    EXPECT_EQ(values[3], *collection.Get("TAG_B", 0));
    EXPECT_EQ(values[4], *collection.Get("TAG_B", 1));
    EXPECT_EQ(values[5], *collection.Get("TAG_C", 0));

    EXPECT_EQ(&values[0], collection.Get("TAG_A", 0));
    EXPECT_EQ(&values[1], collection.Get("TAG_A", 1));
    EXPECT_EQ(&values[2], collection.Get("TAG_A", 2));
    EXPECT_EQ(&values[3], collection.Get("TAG_B", 0));
    EXPECT_EQ(&values[4], collection.Get("TAG_B", 1));
    EXPECT_EQ(&values[5], collection.Get("TAG_C", 0));

    EXPECT_EQ(values[0], *collection_ptr->Get("TAG_A", 0));
    EXPECT_EQ(values[1], *collection_ptr->Get("TAG_A", 1));
    EXPECT_EQ(values[2], *collection_ptr->Get("TAG_A", 2));
    EXPECT_EQ(values[3], *collection_ptr->Get("TAG_B", 0));
    EXPECT_EQ(values[4], *collection_ptr->Get("TAG_B", 1));
    EXPECT_EQ(values[5], *collection_ptr->Get("TAG_C", 0));

    EXPECT_EQ(&values[0], collection_ptr->Get("TAG_A", 0));
    EXPECT_EQ(&values[1], collection_ptr->Get("TAG_A", 1));
    EXPECT_EQ(&values[2], collection_ptr->Get("TAG_A", 2));
    EXPECT_EQ(&values[3], collection_ptr->Get("TAG_B", 0));
    EXPECT_EQ(&values[4], collection_ptr->Get("TAG_B", 1));
    EXPECT_EQ(&values[5], collection_ptr->Get("TAG_C", 0));

    // Test const-ness.
    EXPECT_EQ(false, std::is_const<typename std::remove_reference<
                         decltype(collection.Get("TAG_A", 0))>::type>::value);
    EXPECT_EQ(true,
              std::is_const<typename std::remove_reference<
                  decltype(collection_ptr->Get("TAG_A", 0))>::type>::value);

    // Test access using a range based for.
    int i = 0;
    for (auto& value : *collection_ptr) {
      EXPECT_EQ(&values[i], value);
      EXPECT_EQ(values[i], *value);
      EXPECT_EQ(
          true,
          std::is_const<
              typename std::remove_reference<decltype(value)>::type>::value);
      // In const collections of pointers it's just the (stored) pointer
      // which is const, not the underlying data.
      EXPECT_EQ(
          false,
          std::is_const<
              typename std::remove_reference<decltype(*value)>::type>::value);
      ++i;
    }
    i = 0;
    for (auto& value : collection) {
      EXPECT_EQ(&values[i], value);
      EXPECT_EQ(values[i], *value);
      EXPECT_EQ(
          false,
          std::is_const<
              typename std::remove_reference<decltype(value)>::type>::value);
      EXPECT_EQ(
          false,
          std::is_const<
              typename std::remove_reference<decltype(*value)>::type>::value);
      ++i;
    }

    T injected = inject1;
    collection.Get("TAG_A", 2) = &injected;
    EXPECT_EQ(&injected, collection_ptr->Get("TAG_A", 2));
    EXPECT_EQ(injected, *collection_ptr->Get("TAG_A", 2));
    // Test the random access operator in the iterator.
    // the operator[] should not generally be used.
    EXPECT_EQ(
        &injected,
        collection_ptr->begin()[collection_ptr->GetId("TAG_A", 2).value()]);
    *collection.begin()[collection_ptr->GetId("TAG_A", 2).value()] = inject2;
    EXPECT_EQ(inject2, injected);

    // Test access using a range based for.
    i = 0;
    for (const T* value : *collection_ptr) {
      if (i != collection_ptr->GetId("TAG_A", 2).value()) {
        EXPECT_EQ(&values[i], value);
        EXPECT_EQ(values[i], *value);
      } else {
        EXPECT_EQ(&injected, value);
        EXPECT_EQ(injected, *value);
      }
      ++i;
    }
  }
  return absl::OkStatus();
}

TEST(CollectionTest, TestCollectionWithPointersIntAndString) {
  MP_ASSERT_OK(TestCollectionWithPointers<int>({3, 7, -2, 0, 4, -3}, 17, 10));
  MP_ASSERT_OK(TestCollectionWithPointers<std::string>(
      {"a0", "a1", "a2", "b0", "b1", "c0"}, "inject1", "inject2"));
}

TEST(CollectionTest, TestIteratorFunctions) {
  std::shared_ptr<tool::TagMap> tag_map =
      tool::CreateTagMap({"TAG_A:a", "TAG_B:1:b", "TAG_A:2:c", "TAG_B:d",
                          "TAG_C:0:e", "TAG_A:1:f"})
          .value();

  std::vector<std::string> values = {"a0", "a1", "a2", "b0", "b1", "c0"};
  internal::Collection<std::string, internal::CollectionStorage::kStorePointer>
      collection(tag_map);
  collection.GetPtr(collection.GetId("TAG_A", 0)) = &values[0];
  collection.GetPtr(collection.GetId("TAG_A", 1)) = &values[1];
  collection.GetPtr(collection.GetId("TAG_A", 2)) = &values[2];
  collection.GetPtr(collection.GetId("TAG_B", 0)) = &values[3];
  collection.GetPtr(collection.GetId("TAG_B", 1)) = &values[4];
  collection.GetPtr(collection.GetId("TAG_C", 0)) = &values[5];

  EXPECT_EQ(false, std::is_const<typename std::remove_reference<
                       decltype(collection.begin())>::type>::value);
  EXPECT_EQ(values[0], *collection.begin());
  EXPECT_EQ(false, collection.begin()->empty());
  EXPECT_EQ(false, (*collection.begin()).empty());
  collection.begin()->assign("inject3");
  EXPECT_EQ(values[0], "inject3");

  const auto* collection_ptr = &collection;

  EXPECT_EQ(true, std::is_const<typename std::remove_reference<
                      decltype(*collection_ptr->begin())>::type>::value);
  EXPECT_EQ(values[0], *collection_ptr->begin());
  EXPECT_EQ(false, collection_ptr->begin()->empty());
  EXPECT_EQ(false, (*collection_ptr->begin()).empty());
}

}  // namespace
}  // namespace mediapipe
