from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import io
import stat
import tarfile
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
TOOL_PATH = ROOT / "tools/release/check_python_archives.py"
TOOL_SPEC = importlib.util.spec_from_file_location("check_python_archives", TOOL_PATH)
assert TOOL_SPEC is not None and TOOL_SPEC.loader is not None
ARCHIVE_TOOL = importlib.util.module_from_spec(TOOL_SPEC)
TOOL_SPEC.loader.exec_module(ARCHIVE_TOOL)

ArchiveValidationError = ARCHIVE_TOOL.ArchiveValidationError
main = ARCHIVE_TOOL.main
validate_archives = ARCHIVE_TOOL.validate_archives
validate_sdist = ARCHIVE_TOOL.validate_sdist
validate_wheel = ARCHIVE_TOOL.validate_wheel
safe_member_name = ARCHIVE_TOOL.safe_member_name

RECORD = "pixel4dgs-0.1.dist-info/RECORD"
DEFAULT_FILES = {"p2g/__init__.py": b"release-archive-payload\n"}


def _record_digest(payload: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=")
    return f"sha256={encoded.decode()}"


def _valid_rows(files: Mapping[str, bytes], *, record: str = RECORD) -> list[tuple[str, str, str]]:
    rows = [(name, _record_digest(payload), str(len(payload))) for name, payload in files.items()]
    rows.append((record, "", ""))
    return rows


def _encode_rows(rows: Iterable[Sequence[str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerows(rows)
    return stream.getvalue().encode()


def _zip_info(name: str, mode: int = stat.S_IFREG | 0o644) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name)
    info.create_system = 3
    info.external_attr = mode << 16
    info.compress_type = zipfile.ZIP_STORED
    return info


def _write_wheel(
    path: Path,
    *,
    files: Mapping[str, bytes] = DEFAULT_FILES,
    rows: Iterable[Sequence[str]] | None = None,
    modes: Mapping[str, int] | None = None,
    record: str = RECORD,
) -> Path:
    record_rows = list(rows) if rows is not None else _valid_rows(files, record=record)
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in files.items():
            mode = (modes or {}).get(name, stat.S_IFREG | 0o644)
            archive.writestr(_zip_info(name, mode), payload)
        archive.writestr(_zip_info(record), _encode_rows(record_rows))
    return path


def _write_sdist(
    path: Path,
    members: Iterable[tuple[str, bytes | None, bytes]],
) -> Path:
    with tarfile.open(path, "w:gz") as archive:
        for name, member_type, payload in members:
            info = tarfile.TarInfo(name)
            if member_type is not None:
                info.type = member_type
            if member_type in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                info.linkname = "target"
            if member_type in (tarfile.CHRTYPE, tarfile.BLKTYPE):
                info.devmajor = 1
                info.devminor = 2
            if info.isfile():
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
            else:
                archive.addfile(info)
    return path


def _valid_sdist(path: Path) -> Path:
    return _write_sdist(
        path,
        [
            ("pixel4dgs-0.1/", tarfile.DIRTYPE, b""),
            ("pixel4dgs-0.1/pyproject.toml", tarfile.REGTYPE, b"[project]\n"),
        ],
    )


def _assert_rejected(message: str, operation: Callable[[], None]) -> None:
    with pytest.raises(ArchiveValidationError, match=message):
        operation()


def test_valid_archives_are_accepted_by_library_and_cli(tmp_path: Path) -> None:
    wheel = _write_wheel(tmp_path / "pixel4dgs.whl")
    sdist = _valid_sdist(tmp_path / "pixel4dgs.tar.gz")

    validate_archives(wheels=[wheel], sdists=[sdist])
    assert main(["--wheel", str(wheel), "--sdist", str(sdist)]) == 0


def test_cli_reports_validation_failure_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    wheel = tmp_path / "missing-record.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(_zip_info("p2g/__init__.py"), b"payload")
    sdist = _valid_sdist(tmp_path / "pixel4dgs.tar.gz")

    assert main(["--wheel", str(wheel), "--sdist", str(sdist)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        f"unsafe or invalid archive {wheel}: expected one RECORD file, found 0\n"
    )


def test_validation_never_calls_archive_extraction_apis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wheel = _write_wheel(tmp_path / "pixel4dgs.whl")
    sdist = _valid_sdist(tmp_path / "pixel4dgs.tar.gz")

    def extraction_is_forbidden(*_arguments: object, **_keywords: object) -> None:
        raise AssertionError("archive validator attempted extraction")

    monkeypatch.setattr(zipfile.ZipFile, "extract", extraction_is_forbidden)
    monkeypatch.setattr(zipfile.ZipFile, "extractall", extraction_is_forbidden)
    monkeypatch.setattr(tarfile.TarFile, "extract", extraction_is_forbidden)
    monkeypatch.setattr(tarfile.TarFile, "extractall", extraction_is_forbidden)

    validate_wheel(wheel)
    validate_sdist(sdist)


@pytest.mark.parametrize(
    "member",
    [
        "/absolute.py",
        "../escape.py",
        "package/../../escape.py",
        "package\\escape.py",
        "./package/file.py",
        "package//file.py",
        "package/./file.py",
        ".",
        "C:/escape.py",
        "C:relative.py",
    ],
)
def test_wheel_rejects_unsafe_member_paths(tmp_path: Path, member: str) -> None:
    wheel = _write_wheel(tmp_path / "unsafe.whl", files={member: b"payload"})

    _assert_rejected("unsafe ZIP member path", lambda: validate_wheel(wheel))


def test_wheel_rejects_duplicate_zip_members(tmp_path: Path) -> None:
    wheel = tmp_path / "duplicate.whl"
    rows = _valid_rows(DEFAULT_FILES)
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(_zip_info("p2g/__init__.py"), DEFAULT_FILES["p2g/__init__.py"])
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr(_zip_info("p2g/__init__.py"), b"second payload")
        archive.writestr(_zip_info(RECORD), _encode_rows(rows))

    _assert_rejected("duplicate ZIP member", lambda: validate_wheel(wheel))


def test_wheel_rejects_file_directory_extraction_path_alias(tmp_path: Path) -> None:
    wheel = tmp_path / "aliased.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(_zip_info("package"), b"file")
        archive.writestr(_zip_info("package/", stat.S_IFDIR | 0o755), b"")

    _assert_rejected("extraction-path alias", lambda: validate_wheel(wheel))


def test_wheel_rejects_regular_file_ancestor_conflict(tmp_path: Path) -> None:
    wheel = tmp_path / "ancestor-conflict.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(_zip_info("package"), b"file")
        archive.writestr(_zip_info("package/module.py"), b"module")

    _assert_rejected("regular-file ancestor conflict", lambda: validate_wheel(wheel))


@pytest.mark.parametrize("kind", [stat.S_IFLNK, stat.S_IFCHR, stat.S_IFIFO])
def test_wheel_rejects_non_regular_unix_modes(tmp_path: Path, kind: int) -> None:
    member = "p2g/__init__.py"
    wheel = _write_wheel(
        tmp_path / f"mode-{kind}.whl",
        modes={member: kind | 0o644},
    )

    _assert_rejected("non-regular ZIP member", lambda: validate_wheel(wheel))


@pytest.mark.parametrize(
    "member",
    [
        "/absolute.py",
        "../escape.py",
        "package/../../escape.py",
        "package\\escape.py",
        "./package/file.py",
        "package//file.py",
        "package/./file.py",
        ".",
        "C:/escape.py",
        "C:relative.py",
    ],
)
def test_sdist_rejects_unsafe_member_paths(tmp_path: Path, member: str) -> None:
    sdist = _write_sdist(tmp_path / "unsafe.tar.gz", [(member, tarfile.REGTYPE, b"x")])

    _assert_rejected("unsafe tar member path", lambda: validate_sdist(sdist))


def test_sdist_rejects_duplicate_members(tmp_path: Path) -> None:
    sdist = _write_sdist(
        tmp_path / "duplicate.tar.gz",
        [
            ("package/file.py", tarfile.REGTYPE, b"first"),
            ("package/file.py", tarfile.REGTYPE, b"second"),
        ],
    )

    _assert_rejected("duplicate tar member", lambda: validate_sdist(sdist))


def test_sdist_rejects_file_directory_extraction_path_alias(tmp_path: Path) -> None:
    sdist = _write_sdist(
        tmp_path / "aliased.tar.gz",
        [
            ("package", tarfile.REGTYPE, b"file"),
            ("package/", tarfile.DIRTYPE, b""),
        ],
    )

    _assert_rejected("extraction-path alias", lambda: validate_sdist(sdist))


def test_sdist_rejects_regular_file_ancestor_conflict(tmp_path: Path) -> None:
    sdist = _write_sdist(
        tmp_path / "ancestor-conflict.tar.gz",
        [
            ("package", tarfile.REGTYPE, b"file"),
            ("package/module.py", tarfile.REGTYPE, b"module"),
        ],
    )

    _assert_rejected("regular-file ancestor conflict", lambda: validate_sdist(sdist))


@pytest.mark.parametrize("member", ["", "\x00", "package/\nfile.py", "package/\x7ffile.py"])
def test_member_name_helper_rejects_empty_and_control_characters(member: str) -> None:
    assert not safe_member_name(member)


def test_member_name_helper_allows_one_directory_suffix_only_for_directories() -> None:
    assert safe_member_name("package/file.py")
    assert safe_member_name("package/", directory=True)
    assert not safe_member_name("package/")


@pytest.mark.parametrize(
    "member_type",
    [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE, tarfile.BLKTYPE, tarfile.FIFOTYPE],
)
def test_sdist_rejects_links_devices_and_fifos(tmp_path: Path, member_type: bytes) -> None:
    sdist = _write_sdist(
        tmp_path / f"special-{member_type.hex()}.tar.gz",
        [("package/special", member_type, b"")],
    )

    _assert_rejected("tar contains a non-regular member", lambda: validate_sdist(sdist))


def test_wheel_rejects_missing_record(tmp_path: Path) -> None:
    wheel = tmp_path / "missing-record.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(_zip_info("p2g/__init__.py"), b"payload")

    _assert_rejected("expected one RECORD file, found 0", lambda: validate_wheel(wheel))


def test_wheel_rejects_multiple_record_files(tmp_path: Path) -> None:
    wheel = tmp_path / "multiple-records.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(_zip_info("one.dist-info/RECORD"), b"")
        archive.writestr(_zip_info("two.dist-info/RECORD"), b"")

    _assert_rejected("expected one RECORD file, found 2", lambda: validate_wheel(wheel))


@pytest.mark.parametrize(
    "rows",
    [
        [(RECORD, "", "")],
        [
            ("p2g/__init__.py", _record_digest(DEFAULT_FILES["p2g/__init__.py"]), "24"),
            ("ghost.py", _record_digest(b"ghost"), "5"),
            (RECORD, "", ""),
        ],
    ],
)
def test_wheel_rejects_record_membership_that_is_not_an_exact_file_closure(
    tmp_path: Path, rows: list[tuple[str, str, str]]
) -> None:
    wheel = _write_wheel(tmp_path / "membership.whl", rows=rows)

    _assert_rejected(
        "RECORD membership does not exactly match wheel files", lambda: validate_wheel(wheel)
    )


def test_wheel_rejects_duplicate_record_rows(tmp_path: Path) -> None:
    data = DEFAULT_FILES["p2g/__init__.py"]
    row = ("p2g/__init__.py", _record_digest(data), str(len(data)))
    wheel = _write_wheel(tmp_path / "duplicate-row.whl", rows=[row, row, (RECORD, "", "")])

    _assert_rejected("RECORD contains duplicate paths", lambda: validate_wheel(wheel))


@pytest.mark.parametrize("digest", ["", "md5=deadbeef", "sha256=deadbeef"])
def test_wheel_requires_matching_sha256_for_each_payload(tmp_path: Path, digest: str) -> None:
    name, data = next(iter(DEFAULT_FILES.items()))
    wheel = _write_wheel(
        tmp_path / "digest.whl",
        rows=[(name, digest, str(len(data))), (RECORD, "", "")],
    )
    message = "RECORD digest mismatch" if digest.startswith("sha256=") else "non-sha256"

    _assert_rejected(message, lambda: validate_wheel(wheel))


@pytest.mark.parametrize("declared_size", ["not-an-integer", "0", "25"])
def test_wheel_requires_exact_decimal_record_size(tmp_path: Path, declared_size: str) -> None:
    name, data = next(iter(DEFAULT_FILES.items()))
    wheel = _write_wheel(
        tmp_path / "size.whl",
        rows=[(name, _record_digest(data), declared_size), (RECORD, "", "")],
    )
    message = "invalid RECORD size" if declared_size == "not-an-integer" else "size mismatch"

    _assert_rejected(message, lambda: validate_wheel(wheel))


@pytest.mark.parametrize(
    ("own_digest", "own_size"),
    [("sha256=not-empty", ""), ("", "0"), ("sha256=not-empty", "0")],
)
def test_wheel_requires_empty_hash_and_size_for_record_itself(
    tmp_path: Path, own_digest: str, own_size: str
) -> None:
    name, data = next(iter(DEFAULT_FILES.items()))
    wheel = _write_wheel(
        tmp_path / "record-self.whl",
        rows=[
            (name, _record_digest(data), str(len(data))),
            (RECORD, own_digest, own_size),
        ],
    )

    _assert_rejected("RECORD must leave its own hash and size empty", lambda: validate_wheel(wheel))


def test_wheel_rejects_crc_or_payload_tampering(tmp_path: Path) -> None:
    wheel = _write_wheel(tmp_path / "tampered.whl")
    original = DEFAULT_FILES["p2g/__init__.py"]
    tampered = original.replace(b"payload", b"payloae")
    archive_bytes = wheel.read_bytes()
    assert len(original) == len(tampered)
    assert archive_bytes.count(original) == 1
    wheel.write_bytes(archive_bytes.replace(original, tampered, 1))

    _assert_rejected("CRC failure", lambda: validate_wheel(wheel))
