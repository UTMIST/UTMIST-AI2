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

#

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

# ----------------------------
# Install Tenstorrent SFPI runtime (.deb) AFTER ttnn, then normalize paths
# ----------------------------
SFPI_DEB_URL="https://github.com/tenstorrent/sfpi/releases/download/v6.21.0/sfpi_6.21.0_x86_64.deb"
SFPI_DEB_PATH="/tmp/sfpi_6.21.0_x86_64.deb"

echo "Downloading SFPI package from: $SFPI_DEB_URL"
curl -L "$SFPI_DEB_URL" -o "$SFPI_DEB_PATH"

echo "Installing SFPI package..."
dpkg -i "$SFPI_DEB_PATH" || apt-get -f install -y

# Normalize SFPI location and ensure both expected lookup paths exist without symlink loops
SITE_PKGS="$(python - <<'PY'
import sysconfig
paths = sysconfig.get_paths()
print(paths.get('purelib') or paths.get('platlib') or '')
PY
)"
if [ -n "$SITE_PKGS" ] && [ -d "$SITE_PKGS" ]; then
    echo "Detected site-packages at: $SITE_PKGS"
fi

SFPI_CANON="/opt/tenstorrent/sfpi"
mkdir -p "/opt/tenstorrent"

# Detect where the .deb placed SFPI (common locations)
SFPI_SRC_DIR=""
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

SITE_TTNN_RUNTIME=""
SITE_SFPI=""
if [ -n "$SITE_PKGS" ]; then
    SITE_TTNN_RUNTIME="$SITE_PKGS/ttnn/runtime"
    SITE_SFPI="$SITE_TTNN_RUNTIME/sfpi"
fi

# If only site-packages copy exists, set it as source
if [ -z "$SFPI_SRC_DIR" ] && [ -n "$SITE_SFPI" ] && [ -d "$SITE_SFPI" ]; then
    SFPI_SRC_DIR="$SITE_SFPI"
fi

echo "SFPI source: ${SFPI_SRC_DIR:-<none>}"
echo "SFPI canonical: $SFPI_CANON"
echo "Python runtime target: ${SITE_SFPI:-<none>}"

# If SFPI located inside site-packages, move it to canonical location to avoid symlink loops
if [ -n "$SITE_SFPI" ] && [ "$SFPI_SRC_DIR" = "$SITE_SFPI" ]; then
    if [ -L "$SITE_SFPI" ]; then
        RESOLVED="$(readlink -f "$SITE_SFPI" || true)"
        if [ -n "$RESOLVED" ] && [ -d "$RESOLVED" ]; then
            SFPI_SRC_DIR="$RESOLVED"
        fi
    fi
    if [ "$SFPI_SRC_DIR" = "$SITE_SFPI" ]; then
        rm -rf "$SFPI_CANON"
        mkdir -p "$(dirname "$SFPI_CANON")"
        mv "$SITE_SFPI" "$SFPI_CANON"
        SFPI_SRC_DIR="$SFPI_CANON"
    fi
fi

# Ensure canonical directory has actual files (copy if needed)
if [ "$SFPI_SRC_DIR" != "$SFPI_CANON" ] && [ -n "$SFPI_SRC_DIR" ] && [ -d "$SFPI_SRC_DIR" ]; then
    rm -rf "$SFPI_CANON"
    mkdir -p "$SFPI_CANON"
    cp -a "$SFPI_SRC_DIR/." "$SFPI_CANON/"
fi

# Refresh site-packages runtime link to point to canonical location
if [ -n "$SITE_TTNN_RUNTIME" ]; then
    mkdir -p "$SITE_TTNN_RUNTIME"
    if [ -e "$SITE_SFPI" ] || [ -L "$SITE_SFPI" ]; then
        rm -rf "$SITE_SFPI"
    fi
    ln -s "$SFPI_CANON" "$SITE_SFPI"
    echo "SFPI linked into Python runtime at: $SITE_SFPI -> $SFPI_CANON"
fi

# Verify presence
if [ ! -d "$SFPI_CANON" ]; then
    echo "❌ SFPI canonical directory missing: $SFPI_CANON"
    exit 1
fi
if [ -n "$SITE_SFPI" ] && [ ! -e "$SITE_SFPI/compiler/bin/riscv-tt-elf-g++" ]; then
    echo "❌ SFPI compiler not found at $SITE_SFPI/compiler/bin/riscv-tt-elf-g++"
    echo "Contents of $SITE_SFPI:"
    ls -la "$SITE_SFPI" || true
    echo "Contents of $SFPI_CANON:"
    ls -la "$SFPI_CANON" || true
    exit 1
fi


echo "✅ Installation complete."
