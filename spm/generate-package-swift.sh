#!/usr/bin/env bash
# Generate Package.swift from build checksums
set -e

# Configuration
MPP_BUILD_VERSION="${MPP_BUILD_VERSION:-0.10.26}"
SPM_OUTPUT_DIR="${SPM_OUTPUT_DIR:-./spm/output}"
GITHUB_REPO="${GITHUB_REPO:-mihaidimoiu/mediapipe}"
PACKAGE_SWIFT_PATH="${PACKAGE_SWIFT_PATH:-./Package.swift}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   Package.swift Generator                          ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════╝${NC}"
echo ""

# Get repo root and change to it
MPP_ROOT_DIR=$(git rev-parse --show-toplevel)
cd "$MPP_ROOT_DIR"

# Check if checksums exist
CHECKSUMS_DIR="$SPM_OUTPUT_DIR/checksums"
if [ ! -d "$CHECKSUMS_DIR" ]; then
    echo -e "${RED}❌ Checksums directory not found: $CHECKSUMS_DIR${NC}"
    echo "Run ./spm/build.sh first"
    exit 1
fi

# Convert to absolute paths
SPM_OUTPUT_DIR="$(cd "$SPM_OUTPUT_DIR" && pwd)"
CHECKSUMS_DIR="$SPM_OUTPUT_DIR/checksums"

# Function to get checksum for a framework
get_checksum() {
    local framework_name=$1
    local checksum_file="$CHECKSUMS_DIR/${framework_name}.checksum"

    if [ -f "$checksum_file" ]; then
        cat "$checksum_file"
    else
        echo "CHECKSUM_NOT_FOUND"
    fi
}

# Read checksums
echo -e "${YELLOW}📋 Reading checksums...${NC}"
COMMON_CHECKSUM=$(get_checksum "MediaPipeTasksCommon")
VISION_CHECKSUM=$(get_checksum "MediaPipeTasksVision")
TEXT_CHECKSUM=$(get_checksum "MediaPipeTasksText")
AUDIO_CHECKSUM=$(get_checksum "MediaPipeTasksAudio")

# Validate checksums
if [[ "$COMMON_CHECKSUM" == "CHECKSUM_NOT_FOUND" ]]; then
    echo -e "${RED}❌ Missing checksum for MediaPipeTasksCommon${NC}"
    exit 1
fi

echo -e "${GREEN}✅ All checksums loaded${NC}"
echo ""
echo -e "${YELLOW}📝 Generating Package.swift...${NC}"

# Generate Package.swift
cat > "$PACKAGE_SWIFT_PATH" << EOF
// swift-tools-version: 5.7
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "MediaPipe",
    platforms: [
        .iOS(.v15)
    ],
    products: [
        // Main products - these are what users will import
        .library(
            name: "MediaPipeTasksCommon",
            targets: ["MediaPipeTasksCommon", "MediaPipeTasksCommonWrapper"]),
        .library(
            name: "MediaPipeTasksVision",
            targets: ["MediaPipeTasksVision"]),
        .library(
            name: "MediaPipeTasksText",
            targets: ["MediaPipeTasksText"]),
        .library(
            name: "MediaPipeTasksAudio",
            targets: ["MediaPipeTasksAudio"]),
    ],
    targets: [
        // MediaPipeTasksCommon - Base framework
        // Contains core functionality shared across all task types
        .binaryTarget(
            name: "MediaPipeTasksCommon",
            url: "https://github.com/${GITHUB_REPO}/releases/download/v${MPP_BUILD_VERSION}/MediaPipeTasksCommon.xcframework.zip",
            checksum: "${COMMON_CHECKSUM}"
        ),

        // Wrapper target for MediaPipeTasksCommon to add system framework dependencies
        .target(
            name: "MediaPipeTasksCommonWrapper",
            dependencies: ["MediaPipeTasksCommon"],
            path: "Sources/MediaPipeTasksCommonWrapper",
            linkerSettings: [
                .linkedFramework("Accelerate"),
                .linkedFramework("CoreMedia"),
                .linkedFramework("AssetsLibrary"),
                .linkedFramework("CoreFoundation"),
                .linkedFramework("CoreGraphics"),
                .linkedFramework("CoreImage"),
                .linkedFramework("QuartzCore"),
                .linkedFramework("AVFoundation"),
                .linkedFramework("CoreVideo"),
                .linkedLibrary("c++")
            ]
        ),

        // MediaPipeTasksVision - Vision task APIs
        // Includes: object detection, image classification, face detection, etc.
        .binaryTarget(
            name: "MediaPipeTasksVision",
            url: "https://github.com/${GITHUB_REPO}/releases/download/v${MPP_BUILD_VERSION}/MediaPipeTasksVision.xcframework.zip",
            checksum: "${VISION_CHECKSUM}"
        ),

        // MediaPipeTasksText - Text task APIs
        // Includes: text classification, text embedding, etc.
        .binaryTarget(
            name: "MediaPipeTasksText",
            url: "https://github.com/${GITHUB_REPO}/releases/download/v${MPP_BUILD_VERSION}/MediaPipeTasksText.xcframework.zip",
            checksum: "${TEXT_CHECKSUM}"
        ),

        // MediaPipeTasksAudio - Audio task APIs
        // Includes: audio classification, etc.
        .binaryTarget(
            name: "MediaPipeTasksAudio",
            url: "https://github.com/${GITHUB_REPO}/releases/download/v${MPP_BUILD_VERSION}/MediaPipeTasksAudio.xcframework.zip",
            checksum: "${AUDIO_CHECKSUM}"
        ),
    ]
)
EOF

echo -e "${GREEN}✅ Package.swift generated at: $PACKAGE_SWIFT_PATH${NC}"
echo ""

# Create wrapper source file if it doesn't exist
WRAPPER_DIR="Sources/MediaPipeTasksCommonWrapper"
mkdir -p "$WRAPPER_DIR"

if [ ! -f "$WRAPPER_DIR/dummy.swift" ]; then
    echo -e "${YELLOW}📝 Creating wrapper dummy file...${NC}"
    cat > "$WRAPPER_DIR/dummy.swift" << 'EOF'
// This file is required for SPM to recognize this as a valid target.
// The actual functionality is provided by the MediaPipeTasksCommon binary target.
// This wrapper exists solely to attach linker settings for system frameworks.
EOF
    echo -e "${GREEN}✅ Created $WRAPPER_DIR/dummy.swift${NC}"
fi

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   Package.swift Generated Successfully!            ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}📝 Next Steps:${NC}"
echo "1. Review the generated Package.swift"
echo "2. Test locally: swift package resolve"
echo "3. Commit Package.swift to your repository"
echo "4. Users can now add your package in Xcode!"
echo ""
echo -e "${GREEN}Done! 🎉${NC}"
