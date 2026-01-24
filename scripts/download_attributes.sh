#!/bin/bash
# scripts/download_attributes.sh
# 下载 Market-1501 属性标注文件

set -e

echo "=== Downloading Market-1501 Attribute Annotations ==="

ATTR_DIR="datasets/attributes"
mkdir -p $ATTR_DIR
cd $ATTR_DIR

# 从官方仓库下载
echo "Downloading from GitHub..."
wget -q https://github.com/vana77/Market-1501_Attribute/raw/master/market_attribute.mat -O market1501_attribute.mat

if [ -f "market1501_attribute.mat" ]; then
    echo "✓ Market-1501 属性文件下载完成!"
    echo "  保存位置: $ATTR_DIR/market1501_attribute.mat"
    ls -la market1501_attribute.mat
else
    echo "✗ 下载失败，请手动下载："
    echo "  https://github.com/vana77/Market-1501_Attribute"
    exit 1
fi
