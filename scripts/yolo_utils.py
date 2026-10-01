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
    """Пише data.yaml для Ultralytics і classes.txt (назва класу в рядку = class_id з 0).

    class_id у файлах міток завжди відповідає цьому списку, навіть якщо детектор RoI
    тренується з single_cls і класи ігнорує.

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
    (out_dir / "classes.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8", newline="\n")
    return yaml_path
