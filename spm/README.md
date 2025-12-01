# MediaPipe Swift Package Manager (SPM) Distribution

This directory contains scripts and tools for building and distributing MediaPipe frameworks via Swift Package Manager.

## 📋 Overview

MediaPipe provides the following frameworks for iOS:

- **MediaPipeTasksCommon** - Core functionality and shared components ✅
- **MediaPipeTasksVision** - Computer vision tasks (object detection, face detection, etc.) ✅
- **MediaPipeTasksText** - Text processing tasks (classification, embedding, etc.) ✅
- **MediaPipeTasksAudio** - Audio processing tasks ✅
- **MediaPipeTasksGenAI** - Generative AI capabilities ⚠️ *Currently incomplete for iOS*
- **MediaPipeTasksGenAIC** - Generative AI C APIs ⚠️ *Currently incomplete for iOS*

**Note:** GenAI/GenAIC frameworks are not currently buildable for iOS due to missing source files (`llm_inference_engine_ios.h/cc`). Only the first four frameworks are included in builds.

## ⚙️ Prerequisites & Setup

Before building MediaPipe frameworks, you need to install dependencies and configure your environment.

### 1. Install Homebrew Dependencies

```bash
# Install Bazel
brew install bazelisk

# Install OpenCV 4 (required for macOS builds)
brew install opencv

# Install GitHub CLI (for uploads)
brew install gh
```

### 2. Verify OpenCV 4 Installation

Check your OpenCV version and installation path:

```bash
brew ls opencv | grep version.hpp
```

Expected output (version may vary):
```
/opt/homebrew/Cellar/opencv/4.12.0_15/include/opencv4/opencv2/core/version.hpp
/opt/homebrew/Cellar/opencv/4.12.0_15/include/opencv4/opencv2/dnn/version.hpp
```

### 3. Install Python Dependencies

```bash
# Install setuptools (required for Bazel hermetic Python)
pip3 install setuptools
```

### 4. Configure OpenCV for Your System

Update the OpenCV configuration to match your installation:

**Step 4a: Get your OpenCV version**
```bash
brew ls opencv | grep version.hpp
# Example output: /opt/homebrew/Cellar/opencv/4.12.0_15/include/opencv4/opencv2/core/version.hpp
# Your version: 4.12.0_15
```

**Step 4b: Update WORKSPACE**

Edit `WORKSPACE` at line ~634:
```python
new_local_repository(
    name = "macos_opencv",
    build_file = "@//third_party:opencv_macos.BUILD",
    path = "/opt/homebrew/Cellar",  # or /usr/local/Cellar for Intel Macs
)
```

**Step 4c: Update opencv_macos.BUILD**

Edit `third_party/opencv_macos.BUILD` at line ~38:
```python
PREFIX = "opencv/4.12.0_15"  # Replace with your version from Step 4a
```

And at lines ~54-55:
```python
hdrs = glob([paths.join(PREFIX, "include/opencv4/opencv2/**/*.h*")]),
includes = [paths.join(PREFIX, "include/opencv4")],
```

**Note:** The configuration files are already set up for OpenCV 4. If you need OpenCV 3, see the comments in `third_party/opencv_macos.BUILD` for instructions.

### 5. Verify Bazel Configuration

Ensure your `.bazelrc` has the correct settings:

**For iOS builds** (line ~85):
```bash
build:ios --incompatible_enable_cc_toolchain_resolution
```

### 6. Authentication (for uploads)

If you plan to upload releases:

```bash
gh auth login
```

## 🚀 Quick Start

### Build and Release Process

1. **Build all frameworks:**
   ```bash
   ./spm/build.sh
   ```

   This will:
   - Build all XCFrameworks using Bazel
   - Create SPM-compatible ZIP archives
   - Compute checksums for each framework
   - Generate a checksum report

2. **Generate Package.swift:**
   ```bash
   ./spm/generate-package-swift.sh
   ```

   This creates/updates the root `Package.swift` with the correct checksums and URLs.

3. **Upload to GitHub Releases:**
   ```bash
   # Set your repository (if different from google/mediapipe)
   export GITHUB_REPO="YOUR_ORG/mediapipe"

   ./spm/upload-release.sh
   ```

   This will:
   - Create a new GitHub release (as draft)
   - Upload all framework archives
   - Upload the checksum report

4. **Publish the release:**
   ```bash
   gh release edit v0.10.0 --draft=false
   ```

5. **Commit and tag:**
   ```bash
   git add Package.swift Sources/
   git commit -m "Add SPM support for v0.10.0"
   git tag v0.10.0
   git push origin main --tags
   ```

**Usage:**
```bash
# Use default version
./spm/build.sh

# Specify version
MPP_BUILD_VERSION=0.11.0 ./spm/build.sh

# Custom output directory
SPM_OUTPUT_DIR=/tmp/spm-build ./spm/build.sh
```

### `generate-package-swift.sh`

Generates the `Package.swift` manifest file with correct checksums and URLs.

**Environment Variables:**
- `MPP_BUILD_VERSION` - Version number (default: 0.10.0)
- `GITHUB_REPO` - GitHub repository (default: google/mediapipe)
- `PACKAGE_SWIFT_PATH` - Output path (default: ./Package.swift)
- `SPM_OUTPUT_DIR` - Where to find checksums (default: ./spm/output)

**Usage:**
```bash
# Generate with defaults
./spm/generate-package-swift.sh

# For a fork
GITHUB_REPO="myorg/mediapipe" ./spm/generate-package-swift.sh

# Specific version
MPP_BUILD_VERSION=0.11.0 ./spm/generate-package-swift.sh
```

### `upload-release.sh`

Uploads built frameworks to GitHub Releases using the GitHub CLI (`gh`).

**Prerequisites:**
- Install GitHub CLI: `brew install gh`
- Authenticate: `gh auth login`

**Environment Variables:**
- `MPP_BUILD_VERSION` - Version number (default: 0.10.0)
- `GITHUB_REPO` - GitHub repository (default: google/mediapipe)
- `SPM_OUTPUT_DIR` - Where to find archives (default: ./spm/output)

**Usage:**
```bash
# Upload to google/mediapipe
./spm/upload-release.sh

# Upload to your fork
GITHUB_REPO="myorg/mediapipe" ./spm/upload-release.sh
```

## 🔧 Advanced Usage

### Building a Single Framework

```bash
# Build only MediaPipeTasksVision
FRAMEWORK_NAME=MediaPipeTasksVision \
MPP_BUILD_VERSION=0.10.0 \
DEST_DIR=$HOME/mediapipe-frameworks \
./mediapipe/tasks/ios/build_ios_framework.sh
```

### Custom Release Workflow

```bash
# Set version and repo
export MPP_BUILD_VERSION=0.11.0
export GITHUB_REPO="myorg/mediapipe"

# Build
./spm/build.sh

# Generate Package.swift
./spm/generate-package-swift.sh

# Review checksums
cat spm/output/checksums.txt

# Upload (creates draft)
./spm/upload-release.sh

# Review release in browser
gh release view v${MPP_BUILD_VERSION} --web

# Publish when ready
gh release edit v${MPP_BUILD_VERSION} --draft=false

# Commit and tag
git add Package.swift Sources/
git commit -m "Release v${MPP_BUILD_VERSION}"
git tag v${MPP_BUILD_VERSION}
git push origin main --tags
```

## 📝 For Package Users

Once released, users can add MediaPipe to their projects:

### Using Xcode

1. File → Add Package Dependencies
2. Enter repository URL: `https://github.com/google/mediapipe`
3. Select version `0.10.0` (or desired version)
4. Choose the frameworks you need

### Using Package.swift

```swift
// swift-tools-version: 5.7
import PackageDescription

let package = Package(
    name: "MyApp",
    platforms: [.iOS(.v15)],
    dependencies: [
        .package(url: "https://github.com/google/mediapipe", from: "0.10.0")
    ],
    targets: [
        .target(
            name: "MyApp",
            dependencies: [
                .product(name: "MediaPipeTasksVision", package: "mediapipe"),
                .product(name: "MediaPipeTasksCommon", package: "mediapipe"),
            ]
        )
    ]
)
```

### Build Requirements

- macOS with Xcode 14+
- Bazel (version specified in `.bazelversion`)
- OpenCV 4 (installed via Homebrew)
- Python 3.12+ with setuptools
- Swift 5.7+
- GitHub CLI (`gh`) for uploads

**See the [Prerequisites & Setup](#️-prerequisites--setup) section for detailed installation instructions.**

## 📚 Additional Resources

- [Swift Package Manager Documentation](https://swift.org/package-manager/)
- [Binary Dependencies in SPM](https://developer.apple.com/documentation/xcode/distributing-binary-frameworks-as-swift-packages)
- [MediaPipe Documentation](https://developers.google.com/mediapipe)
- [GitHub CLI Documentation](https://cli.github.com/manual/)

## 🤝 Contributing

When making changes to the SPM distribution:

1. Test the full build process locally
2. Verify Package.swift with `swift package resolve`
3. Test in a sample Xcode project
4. Update this README if adding new scripts or features
5. Document any breaking changes

## 📄 License

See [LICENSE](../LICENSE) file in the root directory.
