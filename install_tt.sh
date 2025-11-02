#!/usr/bin/env bash
set -e

remove_path() {
    if [ -e "$1" ]; then
        echo "Removing existing $1..."
        rm -rf "$1"
    fi
}

# ----------------------------
# Install system packages
# ----------------------------

apt update && apt install -y python3-apt

apt-get update && apt-get install -y ffmpeg

apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0

apt-get update && apt-get install -y \
    unzip \
    python3 \
    python3-dev\
    python3-pip \
    gcc \
    g++ \
    libapt-pkg-dev \
    build-essential

# ----------------------------
# Upgrade pip and install gdown if missing
# ----------------------------
python -m pip install --upgrade pip
if ! python -m pip show gdown &> /dev/null; then
    python -m pip install gdown --no-cache-dir
fi

# ----------------------------
# Download & unzip Google Drive files
# ----------------------------
download_and_unzip() {
    FILE_ID="$1"
    DEST="$2"
    ZIP_NAME="${DEST}.zip"

    remove_path "$DEST"
    remove_path "$ZIP_NAME"

    echo "Downloading $ZIP_NAME..."
    gdown "$FILE_ID" -O "$ZIP_NAME"

    if unzip -tq "$ZIP_NAME" &> /dev/null; then
        echo "Unzipping $ZIP_NAME..."
        unzip -q "$ZIP_NAME"
        rm "$ZIP_NAME"
    else
        echo "❌ $ZIP_NAME is not a valid zip file!"
        exit 1
    fi
}

download_and_unzip 1F2MJQ5enUPVtyi3s410PUuv8LiWr8qCz assets
download_and_unzip 1LAOL8sYCUfsCk3TEA3vvyJCLSl0EdwYB attacks

# ----------------------------
# Install Python dependencies
# ----------------------------
if [ -f requirements.txt ]; then
    echo "Installing Python dependencies..."
    python3 -m pip install -r requirements.txt
else
    echo "No requirements.txt found. Skipping Python dependency installation."
fi

python -m pip install ttnn

python -m pip install numpy==2.1.1

echo "✅ Installation complete."
