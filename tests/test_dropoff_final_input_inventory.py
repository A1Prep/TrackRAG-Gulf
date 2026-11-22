import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tools import dropoff_final_input_inventory as inventory
from tools import dropoff_official_final_scenes as originals
from tools.dropoff_final_handoff_checks import input_overlaps
from tools.dropoff_package_final_scenes import package


class FinalInputInventoryTests(unittest.TestCase):
    def test_only_permitted_input_values_reach_decoder(self):
        hidden = 'sensitive } ] quoted "request": {not a new key}'
        record = {'case_id': 'r', 'reviewed': {'nested': [hidden, {'deeper': hidden}]},
                  'source': {'mask_path': hidden, 'image_sha256': 'reference'}, 'request': ['visible']}
        serialized = json.dumps(record, ensure_ascii=False)
        fields = {'case_id': True, 'source': {'mask_path': 'presence', 'image_sha256': True}, 'request': True}

        class CheckingDecoder:
            def raw_decode(self, data, start):
                item, end = json.JSONDecoder().raw_decode(data, start)
                self_outer.assertNotIn('sensitive', data[start:end])
                return item, end

        self_outer = self
        with patch.object(inventory, 'DECODER', CheckingDecoder()):
            result, end = inventory.project(serialized, fields)
        self.assertEqual(end, len(serialized))
        self.assertEqual(result, {'case_id': 'r', 'request': ['visible'],
                                  'source': {'mask_path': True, 'image_sha256': 'reference'}})

    def test_presence_distinguishes_missing_null_and_supplied(self):
        fields = {'mask_path': 'presence', 'mask_sha256': 'presence'}
        self.assertEqual(inventory.project('{}', fields)[0], {})
        self.assertEqual(inventory.project('{"mask_path":null,"mask_sha256":"hidden"}', fields)[0],
                         {'mask_path': False, 'mask_sha256': True})

    def test_language_overlap_is_kept_separate_from_frame_and_recording_overlap(self):
        def request(recording, text, image):
            return {'request_id': 'case-' + recording, 'text': text,
                    'scene': {'frame': {'recording_id': recording}, 'provenance': {'image_sha256': image * 64}}}

        first = [request('A', '  Stop HERE', 'a'), request('B', 'stop here', 'b')]
        second = [{'request': request('C', 'Stop here', 'c')}]
        result = input_overlaps(first, second)
        self.assertEqual(result['final_rows_with_exact_other_wording'], 2)
        self.assertEqual(result['shared_unique'], {'request_ids': 0, 'recording_ids': 0,
                                                   'normalized_instructions': 1, 'images_sha256': 0})

    def test_bounded_range_reassembly_preserves_every_byte_and_digest(self):
        payload = bytes(range(31))
        fetched = []

        def source(reader, start):
            fetched.append(start)
            return payload[start:start + reader.block]

        with patch.object(originals, 'ARCHIVE_SIZE', len(payload)), patch.object(originals.Ranges, 'fetch', source):
            reader = originals.Ranges(7, 2)
            try:
                self.assertEqual(b''.join(reader.read(n) for n in (0, 1, 11, 4, 9, 5, 200)), payload)
                self.assertEqual(reader.read(4), b'')
                self.assertEqual(reader.processed, len(payload))
                self.assertEqual(reader.sha.hexdigest(), hashlib.sha256(payload).hexdigest())
                self.assertEqual(sorted(fetched), [0, 7, 14, 21, 28])
            finally:
                reader.close()

    def test_canonical_archive_rejects_partial_runs_and_a_different_source(self):
        with TemporaryDirectory() as temp:
            area = Path(temp)
            status = area / 'status.json'
            status.write_text(json.dumps({'state': 'hashing'}))
            with self.assertRaisesRegex(ValueError, 'source exports must finish'):
                package(area, area, area / 'archive.zip', area / 'result.json')
            status.write_text(json.dumps({'state': 'complete', 'no_models_called': True,
                                          'label_files_opened': False, 'full_archive_sha_verified': True,
                                          'archive_sha256': 'different'}))
            with self.assertRaisesRegex(ValueError, 'source SHA'):
                package(area, area, area / 'archive.zip', area / 'result.json')
            self.assertFalse((area / 'archive.zip').exists())


if __name__ == '__main__':
    unittest.main()
