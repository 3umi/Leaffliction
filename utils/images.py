import os

from typing import List

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def is_image(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in IMAGE_EXTS


def list_images(path: str) -> List[str]:
    return sorted(
        e.path for e in os.scandir(path)
        if not e.name.startswith(".")
        and e.is_file(follow_symlinks=False)
        and is_image(e.name)
    )
