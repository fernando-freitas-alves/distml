#! /usr/bin/env bash
# shellcheck disable=SC2086
set -u

APPDIR=$(dirname "$0")/..
DOWNLOAD_DIR="$APPDIR/data/MNIST/raw"
OPTS="-N --retry-on-http-error=503 --tries=5"

pushd "$DOWNLOAD_DIR" || exit 1

wget $OPTS http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget $OPTS http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
wget $OPTS http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget $OPTS http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz

popd || exit
