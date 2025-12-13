#!/usr/bin/env bash
# Build script for creating SPM-compatible MediaPipe frameworks
set -e

# Configuration
MPP_BUILD_VERSION="${MPP_BUILD_VERSION:-0.10.26}"
GITHUB_REPO="${GITHUB_REPO:-mihaidimoiu/mediapipe}"
# Use /tmp by default to avoid repo-root restriction in build_ios_framework.sh
# The original build script doesn't allow DEST_DIR under the repo root
DEST_DIR="${DEST_DIR:-/tmp/mpp-frameworks-build}"
SPM_OUTPUT_DIR="${SPM_OUTPUT_DIR:-./spm/output}"
# Note: GenAI/GenAIC frameworks currently incomplete for iOS (missing llm_inference_engine_ios.h/cc)
# FRAMEWORKS=("MediaPipeTasksCommon" "MediaPipeTasksVision" "MediaPipeTasksText" "MediaPipeTasksAudio" "MediaPipeTasksGenAIC" "MediaPipeTasksGenAI")
FRAMEWORKS=("MediaPipeTasksCommon" "MediaPipeTasksVision" "MediaPipeTasksText" "MediaPipeTasksAudio")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   MediaPipe SPM Framework Builder                  ║${NC}"
echo -e "${GREEN}║   Version: ${MPP_BUILD_VERSION}                    ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════╝${NC}"
echo ""

# Get the root directory of the repo
MPP_ROOT_DIR=$(git rev-parse --show-toplevel)
cd "$MPP_ROOT_DIR"

# Clean and create output directories
echo -e "${YELLOW}📁 Setting up output directories...${NC}"
rm -rf "$DEST_DIR"
rm -rf "$SPM_OUTPUT_DIR"
mkdir -p "$DEST_DIR"
mkdir -p "$SPM_OUTPUT_DIR"
mkdir -p "$SPM_OUTPUT_DIR/archives"
mkdir -p "$SPM_OUTPUT_DIR/checksums"

# Convert to absolute path to avoid issues when changing directories
SPM_OUTPUT_DIR="$(cd "$SPM_OUTPUT_DIR" && pwd)"
DEST_DIR="$(cd "$DEST_DIR" && pwd)"

# Create checksum report file
CHECKSUM_REPORT="$SPM_OUTPUT_DIR/checksums.txt"
echo "MediaPipe Framework Checksums - Version $MPP_BUILD_VERSION" > "$CHECKSUM_REPORT"
echo "Generated: $(date)" >> "$CHECKSUM_REPORT"
echo "========================================" >> "$CHECKSUM_REPORT"
echo "" >> "$CHECKSUM_REPORT"

# Function to build a single framework
build_framework() {
    local framework_name=$1
    echo ""
    echo -e "${GREEN}🔨 Building $framework_name...${NC}"

    FRAMEWORK_NAME="$framework_name" \
    MPP_BUILD_VERSION="$MPP_BUILD_VERSION" \
    DEST_DIR="$DEST_DIR" \
    ARCHIVE_FRAMEWORK=true \
    IS_RELEASE_BUILD=false \
    ./mediapipe/tasks/ios/build_ios_framework.sh

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Successfully built $framework_name${NC}"
    else
        echo -e "${RED}❌ Failed to build $framework_name${NC}"
        return 1
    fi
}

# Function to create SPM-compatible ZIP archive
create_spm_archive() {
    local framework_name=$1
    echo -e "${YELLOW}📦 Creating SPM archive for $framework_name...${NC}"

    local framework_dir="$DEST_DIR/$framework_name/$MPP_BUILD_VERSION"

    if [ ! -d "$framework_dir" ]; then
        echo -e "${RED}❌ Framework directory not found: $framework_dir${NC}"
        return 1
    fi

    # Extract the tar.gz archive created by build script
    local tar_file="$framework_dir/${framework_name}-${MPP_BUILD_VERSION}.tar.gz"
    local temp_extract_dir=$(mktemp -d)

    if [ -f "$tar_file" ]; then
        echo "  Extracting framework from tar.gz..."
        tar -xzf "$tar_file" -C "$temp_extract_dir"

        # Add Info.plist files to framework bundles (required by Xcode)
        echo "  Adding Info.plist files..."
        ./spm/add_info_plists.sh \
            "$temp_extract_dir/frameworks/${framework_name}.xcframework" \
            "$framework_name" \
            "$MPP_BUILD_VERSION"

        # Create ZIP archive (SPM prefers ZIP format)
        local zip_file="$SPM_OUTPUT_DIR/archives/${framework_name}.xcframework.zip"
        echo "  Creating ZIP archive..."

        # SPM expects the .xcframework at the root of the ZIP, not in a subdirectory
        cd "$temp_extract_dir/frameworks"
        zip -r -q "$zip_file" "${framework_name}.xcframework"
        cd "$MPP_ROOT_DIR"

        # Clean up temp directory
        rm -rf "$temp_extract_dir"

        if [ -f "$zip_file" ]; then
            local file_size=$(du -h "$zip_file" | cut -f1)
            echo -e "${GREEN}  ✅ Created ZIP: ${framework_name}.xcframework.zip (${file_size})${NC}"

            # Compute checksum
            compute_checksum "$framework_name"
            return 0
        else
            echo -e "${RED}  ❌ Failed to create ZIP archive${NC}"
            return 1
        fi
    else
        echo -e "${RED}❌ Tar file not found: $tar_file${NC}"
        return 1
    fi
}

# Function to compute and save checksum
compute_checksum() {
    local framework_name=$1
    local zip_file="$SPM_OUTPUT_DIR/archives/${framework_name}.xcframework.zip"

    echo "  Computing checksum..."

    # Compute SHA256 checksum using swift package command
    if command -v swift &> /dev/null; then
        local checksum=$(swift package compute-checksum "$zip_file" 2>/dev/null)

        if [ -n "$checksum" ]; then
            # Save to individual file
            echo "$checksum" > "$SPM_OUTPUT_DIR/checksums/${framework_name}.checksum"

            # Add to report
            echo "$framework_name:" >> "$CHECKSUM_REPORT"
            echo "  checksum: \"$checksum\"" >> "$CHECKSUM_REPORT"
            echo "  url: \"https://github.com/${GITHUB_REPO}/releases/download/v${MPP_BUILD_VERSION}/${framework_name}.xcframework.zip\"" >> "$CHECKSUM_REPORT"
            echo "" >> "$CHECKSUM_REPORT"

            echo -e "${GREEN}  ✅ Checksum: $checksum${NC}"
        else
            echo -e "${RED}  ❌ Failed to compute checksum${NC}"
        fi
    else
        echo -e "${YELLOW}  ⚠️  Swift not found, skipping checksum computation${NC}"
        echo "$framework_name: CHECKSUM_NOT_COMPUTED" >> "$CHECKSUM_REPORT"
        echo "" >> "$CHECKSUM_REPORT"
    fi
}

# Build all frameworks
echo -e "${GREEN}Starting framework builds...${NC}"
echo ""

for framework in "${FRAMEWORKS[@]}"; do
    build_framework "$framework"
done

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   Creating SPM Archives                            ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════╝${NC}"
echo ""

# Create SPM archives for all frameworks
for framework in "${FRAMEWORKS[@]}"; do
    create_spm_archive "$framework"
done

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   Build Summary                                    ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}✅ All frameworks built successfully!${NC}"
echo ""
echo "📂 Output locations:"
echo "   - Build artifacts: $DEST_DIR"
echo "   - SPM archives:    $SPM_OUTPUT_DIR/archives"
echo "   - Checksums:       $SPM_OUTPUT_DIR/checksums"
echo "   - Checksum report: $CHECKSUM_REPORT"
echo ""
echo "📋 Checksum Report Preview:"
echo "----------------------------------------"
cat "$CHECKSUM_REPORT"
echo "----------------------------------------"
echo ""
echo -e "${YELLOW}📝 Next Steps:${NC}"
echo "1. Review the checksums in: $CHECKSUM_REPORT"
echo "2. Upload archives from $SPM_OUTPUT_DIR/archives to GitHub releases"
echo "3. Update Package.swift with the checksums"
echo "4. Tag your release: git tag v${MPP_BUILD_VERSION}"
echo ""
echo -e "${GREEN}Done! 🎉${NC}"