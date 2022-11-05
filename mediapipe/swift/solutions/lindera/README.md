## CocoaPods

### Building Pod zipfile
```shell
bazel build -c opt --config=ios_fat --cxxopt=--std=c++17 --copt=-fembed-bitcode //mediapipe/swift/solutions/lindera:podgen
```

### Pushing Pods

here clspecs is the name of pod specs repository
```shell
pod repo push clspecs LinderaDetection.podspec --skip-import-validation
```
