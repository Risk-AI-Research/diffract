"""Physical RAM available to the current process."""

from __future__ import annotations

import ctypes
import logging
import os
import sys
from functools import cache
from pathlib import Path, PurePosixPath

logger = logging.getLogger(__name__)


def available_host_bytes() -> int | None:
    """Read available RAM, respecting the process's cgroup v2 limits on Linux."""
    try:
        if sys.platform == "darwin":
            return _darwin_available()
        if sys.platform == "win32":
            return _windows_available()
        if sys.platform == "linux":
            return _linux_available()
    except (AttributeError, OSError, ValueError):
        return None
    return None


@cache
def _darwin_host(_pid: int) -> tuple:
    # Mach host capabilities are process-local. The cache key changes on fork.
    lib = ctypes.CDLL(None)
    lib.mach_host_self.restype = ctypes.c_uint32
    statistics = lib.host_statistics64
    statistics.argtypes = (
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
    )
    statistics.restype = ctypes.c_int
    return lib.mach_host_self(), statistics, os.sysconf("SC_PAGESIZE")


def _darwin_available() -> int | None:
    # XNU HOST_VM_INFO64. free_count already includes speculative pages.
    host, statistics, page_size = _darwin_host(os.getpid())
    words = (ctypes.c_uint64 * 32)()
    count = ctypes.c_uint32(ctypes.sizeof(words) // ctypes.sizeof(ctypes.c_uint32))
    if statistics(host, 4, ctypes.byref(words), ctypes.byref(count)) != 0:
        return None
    fields = ctypes.cast(words, ctypes.POINTER(ctypes.c_uint32))
    return int((fields[0] + fields[2]) * page_size)


class _WindowsMemoryStatus(ctypes.Structure):
    _fields_ = (
        ("length", ctypes.c_uint32),
        ("load", ctypes.c_uint32),
        ("total_physical", ctypes.c_uint64),
        ("available_physical", ctypes.c_uint64),
        ("total_page_file", ctypes.c_uint64),
        ("available_page_file", ctypes.c_uint64),
        ("total_virtual", ctypes.c_uint64),
        ("available_virtual", ctypes.c_uint64),
        ("available_extended_virtual", ctypes.c_uint64),
    )


def _windows_available() -> int | None:
    status = _WindowsMemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return int(status.available_physical)


@cache
def _cgroup_v2_directories(pid: int) -> tuple[Path, ...]:
    process = Path(f"/proc/{pid}")
    try:
        group = next(
            PurePosixPath(row.removeprefix("0::"))
            for row in (process / "cgroup").read_text().splitlines()
            if row.startswith("0::")
        )
        mounts = (process / "mountinfo").read_text().splitlines()
    except (OSError, StopIteration):
        return ()
    if not group.is_absolute() or ".." in group.parts:
        return ()
    for line in mounts:
        mount, _, filesystem = line.partition(" - ")
        if not filesystem.startswith("cgroup2 "):
            continue
        mount_fields = mount.split()
        root = PurePosixPath(mount_fields[3])
        base = Path(mount_fields[4])
        relative = (
            group.relative_to(root).parts
            if group.is_relative_to(root)
            else group.parts[1:]
        )
        leaf = base.joinpath(*relative)
        if (leaf / "memory.current").is_file():
            return (
                leaf,
                *(parent for parent in leaf.parents if parent.is_relative_to(base)),
            )
    return ()


def _linux_available() -> int | None:
    with Path("/proc/meminfo").open() as meminfo:
        available = next(
            (
                int(line.split()[1]) * 1024
                for line in meminfo
                if line.startswith("MemAvailable:")
            ),
            None,
        )
    for directory in _cgroup_v2_directories(os.getpid()):
        try:
            limit = (directory / "memory.max").read_text().strip()
            if limit == "max":
                continue
            remaining = max(
                0, int(limit) - int((directory / "memory.current").read_text())
            )
        except (OSError, ValueError):
            logger.debug("Cannot read a cgroup memory limit in %s", directory)
            continue
        available = remaining if available is None else min(available, remaining)
    return available
