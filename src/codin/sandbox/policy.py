from __future__ import annotations

"""Simple sandbox policy helpers used by the CLI debug command."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Union


class SandboxPermission(Enum):
    """Permission identifiers mirroring the Rust enum values."""

    DISK_FULL_READ_ACCESS = "disk-full-read-access"
    DISK_WRITE_PLATFORM_USER_TEMP_FOLDER = "disk-write-platform-user-temp-folder"
    DISK_WRITE_PLATFORM_GLOBAL_TEMP_FOLDER = "disk-write-platform-global-temp-folder"
    DISK_WRITE_CWD = "disk-write-cwd"
    DISK_FULL_WRITE_ACCESS = "disk-full-write-access"
    NETWORK_FULL_ACCESS = "network-full-access"


@dataclass
class SandboxWriteFolderPermission:
    """Permission to write to a specific folder."""

    folder: Path

    def __str__(self) -> str:
        return f"disk-write-folder={self.folder}"


SandboxPermissionType = Union[SandboxPermission, SandboxWriteFolderPermission]


@dataclass
class SandboxPolicy:
    """Collection of sandbox permissions."""

    permissions: List[SandboxPermissionType]

    def __str__(self) -> str:  # pragma: no cover - trivial
        perms = ", ".join(str(p) for p in self.permissions)
        return f"SandboxPolicy([{perms}])"

    @classmethod
    def new_read_only_policy(cls) -> "SandboxPolicy":
        return cls([SandboxPermission.DISK_FULL_READ_ACCESS])

    @classmethod
    def new_full_auto_policy(cls) -> "SandboxPolicy":
        return cls(
            [
                SandboxPermission.DISK_FULL_READ_ACCESS,
                SandboxPermission.DISK_WRITE_PLATFORM_USER_TEMP_FOLDER,
                SandboxPermission.DISK_WRITE_CWD,
            ]
        )


def parse_sandbox_permission(value: str) -> SandboxPermissionType:
    """Parse a single permission string."""

    if value.startswith("disk-write-folder="):
        path = value.split("=", 1)[1]
        if not path:
            raise ValueError("disk-write-folder requires a path")
        return SandboxWriteFolderPermission(Path(path).expanduser().resolve())

    mapping = {p.value: p for p in SandboxPermission}
    try:
        return mapping[value]
    except KeyError as exc:
        raise ValueError(f"Unknown sandbox permission '{value}'") from exc


def create_sandbox_policy(full_auto: bool, permissions: List[str]) -> SandboxPolicy:
    """Create a SandboxPolicy from CLI flags."""

    if full_auto:
        return SandboxPolicy.new_full_auto_policy()

    if not permissions:
        return SandboxPolicy.new_read_only_policy()

    parsed = [parse_sandbox_permission(p) for p in permissions]
    return SandboxPolicy(parsed)


__all__ = [
    "SandboxPermission",
    "SandboxWriteFolderPermission",
    "SandboxPolicy",
    "parse_sandbox_permission",
    "create_sandbox_policy",
]
