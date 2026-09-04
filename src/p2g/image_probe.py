from __future__ import annotations

import struct
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO

from p2g.errors import ContractError

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
JPEG_SOF_MARKERS = frozenset(
    {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
)


@dataclass(frozen=True)
class ImageProbe:
    container: str
    width: int
    height: int
    bit_depth: int
    channel_order: str
    stored_range: str
    declared_transfer: str | None
    declared_primaries: str | None
    declared_matrix: str | None
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_exact(stream: BinaryIO, size: int, *, context: str) -> bytes:
    payload = stream.read(size)
    if len(payload) != size:
        raise ContractError(f"truncated {context}")
    return payload


def _probe_png(stream: BinaryIO) -> ImageProbe:
    color_type: int | None = None
    width: int | None = None
    height: int | None = None
    bit_depth: int | None = None
    chunks: list[str] = []
    srgb_intent: int | None = None
    gamma: float | None = None
    icc_profile_name: str | None = None

    while True:
        raw_length = stream.read(4)
        if not raw_length:
            raise ContractError("PNG ended before IEND")
        if len(raw_length) != 4:
            raise ContractError("truncated PNG chunk length")
        length = struct.unpack(">I", raw_length)[0]
        chunk_type = _read_exact(stream, 4, context="PNG chunk type")
        try:
            chunk_name = chunk_type.decode("ascii")
        except UnicodeDecodeError as exc:
            raise ContractError("PNG chunk type is not ASCII") from exc
        chunks.append(chunk_name)

        inspect_payload = chunk_type in {b"IHDR", b"sRGB", b"gAMA", b"iCCP"}
        if inspect_payload:
            payload = _read_exact(stream, length, context=f"PNG {chunk_name} payload")
        else:
            stream.seek(length, 1)
            payload = b""
        raw_crc = _read_exact(stream, 4, context=f"PNG {chunk_name} CRC")
        if inspect_payload:
            expected_crc = struct.unpack(">I", raw_crc)[0]
            actual_crc = zlib.crc32(chunk_type)
            actual_crc = zlib.crc32(payload, actual_crc) & 0xFFFFFFFF
            if actual_crc != expected_crc:
                raise ContractError(f"PNG {chunk_name} CRC mismatch")

        if chunk_type == b"IHDR":
            if width is not None or length != 13:
                raise ContractError("PNG must contain one 13-byte IHDR")
            (
                parsed_width,
                parsed_height,
                parsed_bit_depth,
                parsed_color_type,
                compression,
                filtering,
                interlace,
            ) = struct.unpack(">IIBBBBB", payload)
            if parsed_width <= 0 or parsed_height <= 0:
                raise ContractError("PNG dimensions must be positive")
            if compression != 0 or filtering != 0 or interlace not in {0, 1}:
                raise ContractError("unsupported PNG header semantics")
            width = parsed_width
            height = parsed_height
            bit_depth = parsed_bit_depth
            color_type = parsed_color_type
        elif chunk_type == b"sRGB":
            if length != 1:
                raise ContractError("PNG sRGB chunk must contain one byte")
            srgb_intent = payload[0]
        elif chunk_type == b"gAMA":
            if length != 4:
                raise ContractError("PNG gAMA chunk must contain four bytes")
            gamma = struct.unpack(">I", payload)[0] / 100000.0
        elif chunk_type == b"iCCP":
            terminator = payload.find(b"\0")
            if terminator <= 0:
                raise ContractError("PNG iCCP profile name is malformed")
            icc_profile_name = payload[:terminator].decode("latin-1")
        elif chunk_type == b"IEND":
            if length != 0:
                raise ContractError("PNG IEND must be empty")
            break

    if width is None or height is None or bit_depth is None or color_type is None:
        raise ContractError("PNG is missing IHDR")
    channel_order = {0: "L", 2: "RGB", 4: "LA", 6: "RGBA"}.get(color_type, "unknown")
    declared_transfer: str | None = None
    declared_primaries: str | None = None
    if srgb_intent is not None:
        declared_transfer = "IEC 61966-2-1 sRGB"
        declared_primaries = "IEC 61966-2-1 sRGB"
    elif gamma is not None:
        declared_transfer = f"PNG gAMA {gamma:.8g}"

    return ImageProbe(
        container="png",
        width=width,
        height=height,
        bit_depth=bit_depth,
        channel_order=channel_order,
        stored_range="full",
        declared_transfer=declared_transfer,
        declared_primaries=declared_primaries,
        declared_matrix=None,
        metadata={
            "png_chunks": chunks,
            "srgb_rendering_intent": srgb_intent,
            "gamma": gamma,
            "icc_profile_name": icc_profile_name,
        },
    )


def _next_jpeg_marker(stream: BinaryIO) -> int:
    while True:
        prefix = stream.read(1)
        if not prefix:
            raise ContractError("JPEG ended before a start-of-frame marker")
        if prefix != b"\xff":
            continue
        marker = _read_exact(stream, 1, context="JPEG marker")[0]
        while marker == 0xFF:
            marker = _read_exact(stream, 1, context="JPEG marker padding")[0]
        if marker != 0x00:
            return marker


def _probe_jpeg(stream: BinaryIO) -> ImageProbe:
    metadata: dict[str, Any] = {"jfif": False, "adobe_transform": None}
    while True:
        marker = _next_jpeg_marker(stream)
        if marker in {0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
            continue
        if marker == 0xDA:
            raise ContractError("JPEG scan began before a start-of-frame marker")
        length = struct.unpack(">H", _read_exact(stream, 2, context="JPEG segment length"))[0]
        if length < 2:
            raise ContractError("JPEG segment length is invalid")
        payload = _read_exact(stream, length - 2, context="JPEG segment")
        if marker == 0xE0 and payload.startswith(b"JFIF\0"):
            metadata["jfif"] = True
        elif marker == 0xEE and payload.startswith(b"Adobe") and len(payload) >= 12:
            metadata["adobe_transform"] = payload[11]
        if marker not in JPEG_SOF_MARKERS:
            continue
        if len(payload) < 6:
            raise ContractError("JPEG start-of-frame segment is truncated")
        bit_depth = payload[0]
        height, width = struct.unpack(">HH", payload[1:5])
        components = payload[5]
        if width <= 0 or height <= 0 or components <= 0:
            raise ContractError("JPEG dimensions and component count must be positive")
        channel_order = {1: "L", 3: "RGB", 4: "RGBA"}.get(components, "unknown")
        metadata["components"] = components
        metadata["sof_marker"] = f"0x{marker:02x}"
        return ImageProbe(
            container="jpeg",
            width=width,
            height=height,
            bit_depth=bit_depth,
            channel_order=channel_order,
            stored_range="full",
            declared_transfer=None,
            declared_primaries=None,
            declared_matrix=None,
            metadata=metadata,
        )


def probe_image(path: Path) -> ImageProbe:
    """Read container-level image facts without decoding pixels."""

    try:
        with path.open("rb") as stream:
            signature = _read_exact(stream, 8, context=f"image signature for {path}")
            if signature == PNG_SIGNATURE:
                return _probe_png(stream)
            if signature[:2] == b"\xff\xd8":
                stream.seek(2)
                return _probe_jpeg(stream)
    except OSError as exc:
        raise ContractError(f"cannot probe image {path}: {exc}") from exc
    raise ContractError(f"unsupported image container: {path}")
