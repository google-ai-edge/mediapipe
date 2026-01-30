#!/usr/bin/env bash
# Add Info.plist files to framework bundles in XCFrameworks
set -e

XCFRAMEWORK_PATH="$1"
FRAMEWORK_NAME="$2"
VERSION="${3:-1.0.0}"

echo "Adding Info.plist files to $FRAMEWORK_NAME..."

# Function to create Info.plist for a framework
create_info_plist() {
    local framework_path=$1
    local info_plist="$framework_path/Info.plist"

    if [ -f "$info_plist" ]; then
        echo "  Info.plist already exists: $framework_path"
        return 0
    fi

    cat > "$info_plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>${FRAMEWORK_NAME}</string>
    <key>CFBundleIdentifier</key>
    <string>com.google.mediapipe.${FRAMEWORK_NAME}</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>${FRAMEWORK_NAME}</string>
    <key>CFBundlePackageType</key>
    <string>FMWK</string>
    <key>CFBundleShortVersionString</key>
    <string>${VERSION}</string>
    <key>CFBundleVersion</key>
    <string>${VERSION}</string>
    <key>CFBundleSupportedPlatforms</key>
    <array>
        <string>iPhoneOS</string>
        <string>iPhoneSimulator</string>
    </array>
    <key>MinimumOSVersion</key>
    <string>15.0</string>
</dict>
</plist>
EOF

    echo "  ✅ Created Info.plist: $framework_path"
}

# Find all .framework directories in the XCFramework
find "$XCFRAMEWORK_PATH" -name "*.framework" -type d | while read framework; do
    create_info_plist "$framework"
done

echo "✅ All Info.plist files added!"
