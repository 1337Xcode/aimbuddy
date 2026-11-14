#!/usr/bin/env bash
set -euo pipefail

SDK_ROOT="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-$HOME/android-sdk}}"
SDKMANAGER="$SDK_ROOT/cmdline-tools/latest/bin/sdkmanager"
TOOLS_ARCHIVE="https://dl.google.com/android/repository/commandlinetools-linux-13114758_latest.zip"

export ANDROID_HOME="$SDK_ROOT"
export ANDROID_SDK_ROOT="$SDK_ROOT"
export PATH="$SDK_ROOT/platform-tools:$SDK_ROOT/cmdline-tools/latest/bin:$PATH"

if [[ ! -x "$SDKMANAGER" ]]; then
  command -v curl >/dev/null || { echo "curl is required to install Android command-line tools." >&2; exit 1; }
  command -v unzip >/dev/null || { echo "unzip is required to install Android command-line tools." >&2; exit 1; }

  temp_dir="$(mktemp -d)"
  trap 'rm -rf "$temp_dir"' EXIT
  curl -fsSL "$TOOLS_ARCHIVE" -o "$temp_dir/command-line-tools.zip"
  unzip -q "$temp_dir/command-line-tools.zip" -d "$temp_dir"
  mkdir -p "$SDK_ROOT/cmdline-tools"
  rm -rf "$SDK_ROOT/cmdline-tools/latest"
  mv "$temp_dir/cmdline-tools" "$SDK_ROOT/cmdline-tools/latest"
fi

license_status=0
yes | "$SDKMANAGER" --sdk_root="$SDK_ROOT" --licenses >/dev/null || license_status=$?
if [[ "$license_status" -ne 0 && "$license_status" -ne 141 ]]; then
  exit "$license_status"
fi
"$SDKMANAGER" --sdk_root="$SDK_ROOT" --install \
  "platform-tools" \
  "platforms;android-35" \
  "build-tools;36.0.0" \
  "ndk;29.0.13113456" \
  "cmake;3.22.1"

echo "Android SDK is ready at $SDK_ROOT"