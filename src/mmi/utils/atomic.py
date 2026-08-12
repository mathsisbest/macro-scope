from contextlib import contextmanager
from pathlib import Path
import os
import tempfile
from collections.abc import Generator

@contextmanager
def atomic_write(dest: Path) -> Generator[Path, None, None]:
    """Write to a temp file and atomically replace dest on success."""
    fd, tmp_path = tempfile.mkstemp(dir=dest.parent, suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_path)
    try:
        yield tmp
        os.replace(tmp, dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
