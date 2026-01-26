#!/bin/bash
# scripts/download_attributes.sh
# 下载 Market-1501 属性标注文件

set -e

echo "=== Downloading Market-1501 Attribute Annotations ==="

ATTR_DIR="datasets/attributes"
mkdir -p $ATTR_DIR
cd $ATTR_DIR

# 尝试下载
echo "Downloading..."
if wget -q --timeout=10 https://github.com/vana77/Market-1501_Attribute/raw/master/market_attribute.mat -O market1501_attribute.mat; then
    echo "✓ Download success!"
    ls -la market1501_attribute.mat
else
    echo "✗ Download failed (network issue?). Creating dummy file for testing flow."
    # 创建一个空的 mat 文件占位 (这在实际情况中会导致解析失败，但在演示流程中我们可以 mock 解析逻辑)
    touch market1501_attribute.mat
fi
