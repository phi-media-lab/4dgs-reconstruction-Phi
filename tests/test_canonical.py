from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from p2g.canonical import (
    canonical_json_bytes,
    content_id,
    read_json,
    sha256_json,
    write_new_json,
)
from p2g.errors import ContractError, OutputExistsError


class CanonicalJsonTests(unittest.TestCase):
    def test_key_order_and_negative_zero_are_canonical(self) -> None:
        first = {"z": -0.0, "a": [2, 1]}
        second = {"a": [2, 1], "z": 0.0}
        self.assertEqual(canonical_json_bytes(first), canonical_json_bytes(second))
        self.assertEqual(sha256_json(first), sha256_json(second))

    def test_non_finite_values_fail_closed(self) -> None:
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaises(ContractError):
                canonical_json_bytes({"value": value})

    def test_content_id_is_namespaced(self) -> None:
        payload = {"camera": "cam000", "frame": 1}
        self.assertNotEqual(content_id("observation", payload), content_id("track", payload))
        self.assertEqual(content_id("observation", payload), content_id("observation", payload))

    def test_new_json_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.json"
            write_new_json(path, {"value": 1})
            self.assertEqual(read_json(path), {"value": 1})
            with self.assertRaises(OutputExistsError):
                write_new_json(path, {"value": 2})
            self.assertEqual(json.loads(path.read_text()), {"value": 1})


if __name__ == "__main__":
    unittest.main()
