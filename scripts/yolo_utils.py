"""Спільні функції для підготовки датасетів у форматі Ultralytics YOLO."""

import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def reset_split_dirs(out_dir: Path, splits):
    """Створює порожні images/<split> та labels/<split>, видаляючи попередній результат."""
    for kind in ("images", "labels"):
        shutil.rmtree(out_dir / kind, ignore_errors=True)
        for split in splits:
            (out_dir / kind / split).mkdir(parents=True)


def write_data_yaml(out_dir: Path, class_names, splits):
    """Пише data.yaml для Ultralytics.

    Шлях до датасету абсолютний: відносний Ultralytics шукав би у власній
    datasets_dir, а не відносно репозиторію. Після перенесення даних (напр. у WSL)
    скрипт треба запустити ще раз.
    """
    yaml_path = out_dir / "data.yaml"
    lines = [
        "# Згенеровано автоматично, не редагувати вручну",
        f"path: {out_dir.resolve().as_posix()}",
    ]
    lines += [f"{split}: images/{split}" for split in splits]
    lines.append("names:")
    lines += [f"  {idx}: {name}" for idx, name in enumerate(class_names)]
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return yaml_path
