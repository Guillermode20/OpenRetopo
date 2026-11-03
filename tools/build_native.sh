#!/usr/bin/env bash
# Build the OpenRetopo native library for the current platform and copy it
# into the extension folder under open_retopo/native/<platform>/.
set -euo pipefail

cd "$(dirname "$0")/.."

cargo build --release -p openretopo-ffi

case "$(uname -s)-$(uname -m)" in
  Linux-x86_64)  dest="open_retopo/native/linux_x86_64";  src="target/release/libopenretopo.so" ;;
  Linux-aarch64) dest="open_retopo/native/linux_arm64";   src="target/release/libopenretopo.so"; mkdir -p "$dest" ;;
  Darwin-x86_64 | Darwin-arm64)
                dest="open_retopo/native/macos_universal2"; src="target/release/libopenretopo.dylib" ;;
  MINGW*|MSYS*|CYGWIN*)
                dest="open_retopo/native/win_x86_64";      src="target/release/openretopo.dll" ;;
  *) echo "unsupported platform: $(uname -s)-$(uname -m)" >&2; exit 1 ;;
esac

mkdir -p "$dest"
cp "$src" "$dest/"
echo "copied $src -> $dest/"
