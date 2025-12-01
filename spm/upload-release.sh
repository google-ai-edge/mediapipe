#!/usr/bin/env bash
# Upload built frameworks to GitHub releases
set -e

# Configuration
MPP_BUILD_VERSION="${MPP_BUILD_VERSION:-0.10.26}"
SPM_OUTPUT_DIR="${SPM_OUTPUT_DIR:-./spm/output}"
GITHUB_REPO="${GITHUB_REPO:-mihaidimoiu/mediapipe}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   MediaPipe GitHub Release Uploader                ║${NC}"
echo -e "${GREEN}║   Version: ${MPP_BUILD_VERSION}                    ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════╝${NC}"
echo ""

# Get repo root and change to it
MPP_ROOT_DIR=$(git rev-parse --show-toplevel)
cd "$MPP_ROOT_DIR"

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo -e "${RED}❌ GitHub CLI (gh) is not installed${NC}"
    echo "Install it from: https://cli.github.com/"
    exit 1
fi

# Check if user is authenticated
if ! gh auth status &> /dev/null; then
    echo -e "${RED}❌ Not authenticated with GitHub CLI${NC}"
    echo "Run: gh auth login"
    exit 1
fi

# Check if archives exist
if [ ! -d "$SPM_OUTPUT_DIR/archives" ]; then
    echo -e "${RED}❌ Archives directory not found: $SPM_OUTPUT_DIR/archives${NC}"
    echo "Run ./spm/build.sh first"
    exit 1
fi

# Convert to absolute path
SPM_OUTPUT_DIR="$(cd "$SPM_OUTPUT_DIR" && pwd)"

archive_count=$(ls -1 "$SPM_OUTPUT_DIR/archives"/*.zip 2>/dev/null | wc -l)
if [ "$archive_count" -eq 0 ]; then
    echo -e "${RED}❌ No archives found in $SPM_OUTPUT_DIR/archives${NC}"
    echo "Run ./spm/build.sh first"
    exit 1
fi

echo -e "${YELLOW}📦 Found $archive_count framework archives${NC}"
echo ""

# Check if release exists
TAG_NAME="v${MPP_BUILD_VERSION}"
if gh release view "$TAG_NAME" --repo "$GITHUB_REPO" &> /dev/null; then
    echo -e "${YELLOW}⚠️  Release $TAG_NAME already exists${NC}"
    read -p "Do you want to delete and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}🗑️  Deleting existing release...${NC}"
        gh release delete "$TAG_NAME" --repo "$GITHUB_REPO" --yes
    else
        echo -e "${YELLOW}Uploading to existing release...${NC}"
    fi
fi

# Create release if it doesn't exist
if ! gh release view "$TAG_NAME" --repo "$GITHUB_REPO" &> /dev/null; then
    echo -e "${GREEN}📝 Creating release $TAG_NAME...${NC}"

    RELEASE_NOTES="# MediaPipe Tasks v${MPP_BUILD_VERSION}

## Swift Package Manager

This release includes XCFrameworks for Swift Package Manager distribution.

### Frameworks included:
- MediaPipeTasksCommon
- MediaPipeTasksVision
- MediaPipeTasksText
- MediaPipeTasksAudio

### Installation

Add to your \`Package.swift\`:

\`\`\`swift
dependencies: [
    .package(url: \"https://github.com/${GITHUB_REPO}\", from: \"${MPP_BUILD_VERSION}\")
]
\`\`\`

Or in Xcode:
1. File → Add Package Dependencies
2. Enter: \`https://github.com/${GITHUB_REPO}\`
3. Select version \`${MPP_BUILD_VERSION}\`

## Checksums

See the checksums.txt file for framework checksums.
"

    gh release create "$TAG_NAME" \
        --repo "$GITHUB_REPO" \
        --title "MediaPipe Tasks v${MPP_BUILD_VERSION}" \
        --notes "$RELEASE_NOTES" \
        --draft

    echo -e "${GREEN}✅ Release created as draft${NC}"
fi

# Upload archives
echo ""
echo -e "${GREEN}📤 Uploading framework archives...${NC}"
echo ""

for archive in "$SPM_OUTPUT_DIR/archives"/*.zip; do
    filename=$(basename "$archive")
    echo -e "${YELLOW}  Uploading $filename...${NC}"

    gh release upload "$TAG_NAME" "$archive" \
        --repo "$GITHUB_REPO" \
        --clobber

    echo -e "${GREEN}  ✅ Uploaded $filename${NC}"
done

# Upload checksum report
if [ -f "$SPM_OUTPUT_DIR/checksums.txt" ]; then
    echo ""
    echo -e "${YELLOW}📋 Uploading checksums.txt...${NC}"
    gh release upload "$TAG_NAME" "$SPM_OUTPUT_DIR/checksums.txt" \
        --repo "$GITHUB_REPO" \
        --clobber
    echo -e "${GREEN}✅ Uploaded checksums.txt${NC}"
fi

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   Upload Complete!                                 ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}📝 Next Steps:${NC}"
echo "1. Review the draft release:"
echo "   gh release view $TAG_NAME --repo $GITHUB_REPO --web"
echo ""
echo "2. If everything looks good, publish the release:"
echo "   gh release edit $TAG_NAME --repo $GITHUB_REPO --draft=false"
echo ""
echo "3. Update Package.swift with the checksums from:"
echo "   $SPM_OUTPUT_DIR/checksums.txt"
echo ""
echo -e "${GREEN}Done! 🎉${NC}"
