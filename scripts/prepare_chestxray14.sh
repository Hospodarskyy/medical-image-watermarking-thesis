#!/usr/bin/env bash
# Готує ChestX-ray14 до тренування YOLO: розпаковує ще не розпаковані архіви
# з data/chestxray14/raw/archives/ і перезбирає data/chestxray14/yolo/ (разом з data.yaml).
#
# Використання (Git Bash або WSL, з будь-якої папки):
#     bash scripts/prepare_chestxray14.sh
#     PYTHON=.venv/bin/python bash scripts/prepare_chestxray14.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
RAW_DIR="data/chestxray14/raw"
PYTHON="${PYTHON:-python}"
export PYTHONIOENCODING=utf-8

# Розпакований архів позначається файлом <архів>.extracted, щоб не розпаковувати його повторно.
shopt -s nullglob
for archive in "$RAW_DIR"/archives/images_*.tar.gz; do
    marker="$archive.extracted"
    if [[ -f "$marker" ]]; then
        continue
    fi
    echo "Розпаковую $(basename "$archive")..."
    tar -xzf "$archive" -C "$RAW_DIR"
    touch "$marker"
done

if [[ ! -d "$RAW_DIR/images" ]]; then
    echo "Помилка: немає $RAW_DIR/images — покладіть архіви images_0XX.tar.gz у $RAW_DIR/archives/" >&2
    exit 1
fi
echo "Знімків у $RAW_DIR/images: $(find "$RAW_DIR/images" -name '*.png' | wc -l)"

"$PYTHON" scripts/convert_chestxray_bbox_to_yolo.py
