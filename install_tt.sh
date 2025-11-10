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
    curl \
    gcc \
    g++ \
    libapt-pkg-dev \
    build-essential

# ----------------------------
# Install Tenstorrent SFPI runtime (.deb)
# ----------------------------
SFPI_DEB_URL="https://github.com/tenstorrent/sfpi/releases/download/v6.21.0/sfpi_6.21.0_x86_64.deb"
SFPI_DEB_PATH="/tmp/sfpi_6.21.0_x86_64.deb"

echo "Downloading SFPI package from: $SFPI_DEB_URL"
curl -L "$SFPI_DEB_URL" -o "$SFPI_DEB_PATH"

echo "Installing SFPI package..."
# Use dpkg then fix any missing dependencies
dpkg -i "$SFPI_DEB_PATH" || apt-get -f install -y
# Optionally, re-run dpkg if needed (usually not required after -f install)
# dpkg -i "$SFPI_DEB_PATH" || true

# Normalize SFPI location and ensure both expected lookup paths exist
SFPI_DST_DIR="/opt/tenstorrent/sfpi"
mkdir -p "/opt/tenstorrent"

# Detect where the .deb placed SFPI (common locations)
for CAND in \
    "/opt/tenstorrent/sfpi" \
    "/usr/local/lib/ttnn/runtime/sfpi" \
    "/usr/local/lib/sfpi" \
    "/usr/lib/ttnn/runtime/sfpi" \
    "/usr/lib/sfpi" \
    "/usr/local/tenstorrent/sfpi" \
    "/usr/share/sfpi" \
    "/usr/local/share/sfpi"; do
    if [ -d "$CAND" ]; then
        SFPI_SRC_DIR="$CAND"
        break
    fi
done

if [ -z "${SFPI_SRC_DIR:-}" ]; then
    echo "Warning: Could not locate installed SFPI directory; proceeding to create ${SFPI_DST_DIR} if missing."
    mkdir -p "$SFPI_DST_DIR"
else
    # Ensure /opt/tenstorrent/sfpi exists, prefer symlink to canonicalize location
    if [ -e "$SFPI_DST_DIR" ] && [ ! -L "$SFPI_DST_DIR" ]; then
        rm -rf "$SFPI_DST_DIR"
    fi
    if [ ! -e "$SFPI_DST_DIR" ]; then
        ln -s "$SFPI_SRC_DIR" "$SFPI_DST_DIR"
    fi
fi

# Create Python site-packages runtime link: ttnn/runtime/sfpi -> /opt/tenstorrent/sfpi
SITE_PKGS="$(python - <<'PY'
import sysconfig
paths = sysconfig.get_paths()
print(paths.get('purelib') or paths.get('platlib') or '')
PY
)"
if [ -n "$SITE_PKGS" ] && [ -d "$SITE_PKGS" ]; then
    TTNN_RUNTIME_DIR="$SITE_PKGS/ttnn/runtime"
    mkdir -p "$TTNN_RUNTIME_DIR"
    TARGET_LINK="$TTNN_RUNTIME_DIR/sfpi"
    if [ -e "$TARGET_LINK" ] && [ ! -L "$TARGET_LINK" ]; then
        rm -rf "$TARGET_LINK"
    fi
    if [ ! -e "$TARGET_LINK" ]; then
        ln -s "$SFPI_DST_DIR" "$TARGET_LINK"
    fi
    echo "SFPI linked into Python runtime at: $TARGET_LINK"
else
    echo "Warning: Could not determine Python site-packages to link SFPI runtime."
fi

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
