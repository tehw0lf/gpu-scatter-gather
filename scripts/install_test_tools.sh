#!/bin/bash
# Install external wordlist generators for cross-validation testing
#
# This script compiles maskprocessor and hashcat from source to ensure:
# 1. We have the latest versions
# 2. GPU support is properly enabled
# 3. We document exact versions used for validation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="${SCRIPT_DIR}/../tools"

mkdir -p "${TOOLS_DIR}"
cd "${TOOLS_DIR}"

echo "========================================="
echo "Installing External Validation Tools"
echo "========================================="
echo

# Install maskprocessor
echo "📦 Installing maskprocessor..."
echo

if [ -d "maskprocessor" ]; then
    echo "maskprocessor directory exists, pulling latest..."
    cd maskprocessor
    git pull
else
    echo "Cloning maskprocessor..."
    git clone https://github.com/hashcat/maskprocessor.git
    cd maskprocessor
fi

echo "Building maskprocessor..."
cd src
make clean || true
make
cd ..

MASKPROCESSOR_VERSION=$(git describe --tags --always)
echo "✅ maskprocessor built successfully!"
echo "   Version: ${MASKPROCESSOR_VERSION}"
echo "   Path: $(pwd)/src/mp64.bin"
echo

cd "${TOOLS_DIR}"

# Install hashcat
echo "📦 Installing hashcat..."
echo

if [ -d "hashcat" ]; then
    echo "hashcat directory exists, pulling latest..."
    cd hashcat
    git pull
else
    echo "Cloning hashcat..."
    git clone https://github.com/hashcat/hashcat.git
    cd hashcat
fi

echo "Building hashcat..."
make clean || true
make

HASHCAT_VERSION=$(git describe --tags --always)
echo "✅ hashcat built successfully!"
echo "   Version: ${HASHCAT_VERSION}"
echo "   Path: $(pwd)/hashcat"
echo

# Create version info file
cd "${TOOLS_DIR}"
cat > VERSIONS.txt <<EOF
External Tools for Cross-Validation
====================================

Built on: $(date)
System: $(uname -a)

maskprocessor:
  Repository: https://github.com/hashcat/maskprocessor
  Version: ${MASKPROCESSOR_VERSION}
  Binary: ${TOOLS_DIR}/maskprocessor/src/mp64.bin

hashcat:
  Repository: https://github.com/hashcat/hashcat
  Version: ${HASHCAT_VERSION}
  Binary: ${TOOLS_DIR}/hashcat/hashcat

Notes:
- Both tools compiled from source for reproducibility
- Used for cross-validation to ensure correctness
- GPU support enabled (if CUDA available)
EOF

echo
echo "========================================="
echo "Installation Complete!"
echo "========================================="
echo
echo "Tools installed in: ${TOOLS_DIR}"
echo
echo "To use these tools, add them to your PATH:"
echo "  export PATH=\"${TOOLS_DIR}/maskprocessor:\${PATH}\""
echo "  export PATH=\"${TOOLS_DIR}/hashcat:\${PATH}\""
echo
echo "Or create symlinks (recommended):"
echo "  mkdir -p ~/.local/bin"
echo "  ln -s ${TOOLS_DIR}/maskprocessor/src/mp64.bin ~/.local/bin/maskprocessor"
echo "  ln -s ${TOOLS_DIR}/hashcat/hashcat ~/.local/bin/hashcat"
echo
echo "Version information saved to: ${TOOLS_DIR}/VERSIONS.txt"
echo

cat "${TOOLS_DIR}/VERSIONS.txt"
