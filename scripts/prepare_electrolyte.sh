#!/bin/bash
# Run once from the scripts/ directory on Perlmutter to download and split the electrolyte dataset.
# Usage: bash prepare_electrolyte.sh

set -e

mkdir -p electrolyte_data/train/raw electrolyte_data/test/raw

echo "Downloading electrolyte dataset..."
curl -L https://raw.githubusercontent.com/BingqingCheng/cace-lr-fit/main/fit-electrolyte/electrolyte.xyz.zip \
    -o electrolyte_data/electrolyte.xyz.zip

echo "Unzipping..."
unzip -p electrolyte_data/electrolyte.xyz.zip > electrolyte_data/electrolyte.xyz

echo "Splitting into train/test..."
$SCRATCH/code/les/nnpackages/newtonnet/bin/python - <<'EOF'
with open('electrolyte_data/electrolyte.xyz') as f:
    lines = f.readlines()

frames = []
i = 0
while i < len(lines):
    n = int(lines[i].strip())
    frames.append(lines[i:i+n+2])
    i += n + 2

n_train = 1600
train_frames = frames[:n_train]
test_frames  = frames[n_train:]

with open('electrolyte_data/train/raw/electrolyte_train.xyz', 'w') as f:
    for frame in train_frames:
        f.writelines(frame)

with open('electrolyte_data/test/raw/electrolyte_test.xyz', 'w') as f:
    for frame in test_frames:
        f.writelines(frame)

print(f'train frames: {len(train_frames)}')
print(f'test  frames: {len(test_frames)}')
EOF

echo "Done. Data ready in electrolyte_data/"
