#!/usr/bin/env bash
# Fetch the SO-101 URDF + STL meshes from the official SO-ARM100 repo into
# so101_port/urdf/ (this directory is gitignored — meshes are not vendored).
#
# Usage:  bash so101_port/fetch_urdf.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$HERE/urdf"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "[fetch_urdf] cloning TheRobotStudio/SO-ARM100 ..."
git clone --depth 1 https://github.com/TheRobotStudio/SO-ARM100 "$TMP/SO-ARM100"

SRC="$TMP/SO-ARM100/Simulation/SO101"
mkdir -p "$DEST"
cp "$SRC/so101_new_calib.urdf" "$DEST/so101.urdf"
cp -r "$SRC/assets" "$DEST/assets"

echo "[fetch_urdf] staged:"
echo "  $DEST/so101.urdf"
echo "  $DEST/assets/  ($(ls "$DEST/assets"/*.stl | wc -l) STL meshes)"
echo "[fetch_urdf] next: python so101_port/convert_so101_urdf.py"
