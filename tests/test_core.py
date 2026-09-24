import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import espyolo
from espstream import RobotCommandClient


class CoreBehaviorTests(unittest.TestCase):
    def test_iou_uses_both_box_areas(self):
        first = espyolo.FaceBox(0, 0, 10, 10, 1.0)
        second = espyolo.FaceBox(5, 5, 20, 20, 1.0)
        self.assertAlmostEqual(first.io_union(second), 25 / 475)

    def test_memory_round_trip_is_json_safe_and_keeps_mean(self):
        memory = espyolo.FaceMemory(
            name="Ada", mean_embedding=np.array([1, 2, 3], dtype=np.float32)
        )
        memory.add_sample_embedding(np.array([1, 2, 3], dtype=np.float32))
        data = memory.to_dict()
        json.dumps(data)
        restored = espyolo.FaceMemory.from_dict(data)
        np.testing.assert_array_equal(restored.mean_embedding, memory.mean_embedding)
        self.assertEqual(restored.sample_count, 1)
        self.assertEqual(len(restored.best_embeddings), 1)

    def test_memory_dimension_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            espyolo.FaceMemory.from_dict({
                "name": "bad",
                "embedding_dim": 3,
                "mean_embedding": [1, 2],
                "best_embeddings": [],
            })

    def test_memory_file_loader_skips_malformed_json_record(self):
        with tempfile.TemporaryDirectory() as directory:
            old_dir = espyolo.MEMORY_DIR
            espyolo.MEMORY_DIR = Path(directory)
            try:
                (Path(directory) / "broken.json").write_text(
                    '{"name":"broken","embedding_dim":2,"mean_embedding":[1],'
                    '"best_embeddings":[]}'
                )
                self.assertIsNone(espyolo._load_memory_file("missing"))
                self.assertIsNotNone(espyolo._load_memory_file("broken"))
            finally:
                espyolo.MEMORY_DIR = old_dir

    def test_sample_count_increments_once_per_sample(self):
        memory = espyolo.FaceMemory(name="Ada", mean_embedding=np.zeros(3))
        memory.add_sample_embedding(np.ones(3))
        memory.add_sample_embedding(np.ones(3))
        self.assertEqual(memory.sample_count, 2)

    def test_command_frame_is_crlf_terminated(self):
        self.assertEqual(
            RobotCommandClient.format_payload("w", 150), b"w, 150\r\n"
        )

    def test_stop_command_is_forced_after_duplicate_movement(self):
        client = RobotCommandClient()
        with patch("espstream.socket.create_connection") as connect:
            sock = connect.return_value.__enter__.return_value
            self.assertTrue(client.send("w", 100)[0])
            self.assertTrue(client.stop()[0])
            self.assertEqual(sock.sendall.call_args.args[-1], b"p, 0\r\n")


if __name__ == "__main__":
    unittest.main()
