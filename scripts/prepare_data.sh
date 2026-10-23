#!/usr/bin/env bash
# Prepare datasets/ in the layout expected by configs/include/dataset.yml:
#   datasets/mscoco/images/{train2017,val2017}/
#   datasets/mscoco/annotations/{instances_val2017,person_keypoints_val2017}.json
#   datasets/ContactHands/{JPEGImages,eval_Annotations}/     (evaluation only)
#   datasets/singlequery_bhoi_train.json                     (training annotations)
#
# usage: bash scripts/prepare_data.sh [--no-train-images]
#   CONTACTHANDS_DIR=/path/to/ContactHands   symlinked to datasets/ContactHands (download it from the
#                                            official ContactHands release; not hosted here)
#   COCO_DIR=/path/to/existing/coco          symlinked to datasets/mscoco instead of downloading
set -euo pipefail
cd "$(dirname "$0")/.."
ANN_URL="${ANN_URL:-https://github.com/LGECTO-AIL-VIL/SingleQuery-BHOI-Dataset/raw/main/annotations/singlequery_bhoi_train.zip}"   # SingleQuery-BHOI-Dataset repo
ANN_SHA256=6312fbaafe58bf7e29a810819970388ace79497cd59c925e6aae4154ff9772c3
COCO=http://images.cocodataset.org
mkdir -p datasets && cd datasets

if [ -n "${COCO_DIR:-}" ]; then
  ln -sfn "$COCO_DIR" mscoco
else
  mkdir -p mscoco/images && cd mscoco
  wget -c $COCO/annotations/annotations_trainval2017.zip && unzip -qo annotations_trainval2017.zip
  wget -c $COCO/zips/val2017.zip && unzip -qo val2017.zip -d images
  [ "${1:-}" = "--no-train-images" ] || { wget -c $COCO/zips/train2017.zip && unzip -qo train2017.zip -d images; }
  cd ..
fi

[ -z "${CONTACTHANDS_DIR:-}" ] || ln -sfn "$CONTACTHANDS_DIR" ContactHands

wget -c -O singlequery_bhoi_train.zip "$ANN_URL"
echo "$ANN_SHA256  singlequery_bhoi_train.zip" | sha256sum -c -
unzip -qo singlequery_bhoi_train.zip
echo "done: $(pwd)"
