"""Synthetic evaluator checks; no detector, tracker or model is executed.

The optional scoring dependencies are installed in .venv-models for these checks.
The ordinary reviewer environment can collect this file without those packages.
"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from evaluation.run_audit import fingerprint
from evaluation.vision_baseline import evaluate


HAS_METRICS=all(importlib.util.find_spec(name) is not None for name in ('motmetrics','pycocotools'))


@unittest.skipUnless(HAS_METRICS,'Optional motmetrics/pycocotools scoring dependencies are required')
class PretrainedVisionReviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.data=self.root/'data'
        self.run=self.root/'run'
        self.run.mkdir()
        images=self.data/'images/train/rec'
        images.mkdir(parents=True)
        annotations=self.data/'annotations/train'
        annotations.mkdir(parents=True)
        self.labels=annotations/'rec.txt'
        self.labels.write_text(''.join(
            f'{frame} 7 Car 0 0 0 1 2 11 12 0 0 0 0 0 0 0\n' for frame in (1,2)))
        self.detections=[]
        self.tracks=[]
        for frame in (1,2):
            image=images/f'{frame:05d}.png'
            Image.new('RGB',(32,24),(20*frame,70,90)).save(image)
            self.detections.append({'recording_id':'rec','frame_id':frame,'width':32,'height':24,
                'image_path':str(image),'image_sha256':fingerprint(image),'detections':[
                    {'bbox_xyxy':[1,2,11,12],'confidence':0.9,'class_id':3,'class_name':'car'}]})
            self.tracks.append({'recording_id':'rec','frame_id':frame,'track_id':42,
                'source':'predicted','class_name':'car','bbox_xyxy':[1,2,11,12]})
        self.info={'recording_id':'rec','frame_count':2,'first_frame':1,'last_frame':2,
            'native_class_mapping':{'3':'car'},'detections_path':str(self.run/'detections.jsonl'),
            'trackers':[{'gmc_method':'none','tracks_path':str(self.run/'tracks.jsonl')}]}
        self.save()

    def save(self):
        for key,rows in (('detections_path',self.detections),):
            path=Path(self.info[key])
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            self.info['detections_sha256']=fingerprint(path)
        tracker=self.info['trackers'][0]
        path=Path(tracker['tracks_path'])
        path.write_text(''.join(json.dumps(row)+'\n' for row in self.tracks))
        tracker['tracks_sha256']=fingerprint(path)
        (self.run/'summary.json').write_text(json.dumps({'threshold':0.05,'recordings':[self.info]}))

    def test_perfect_coarse_detection_and_identity_metrics(self):
        result=evaluate(self.run,self.data)
        self.assertEqual(result['frame_count'],2)
        self.assertAlmostEqual(result['coco_metrics']['AP'],1.0)
        metrics=result['tracking_metrics']['none']['OVERALL']
        self.assertAlmostEqual(metrics['mota'],1.0)
        self.assertAlmostEqual(metrics['idf1'],1.0)
        self.assertAlmostEqual(metrics['motp'],0.0)
        self.assertEqual(result['metric_policy']['coco_max_detections_per_image'],[1,10,100])

    def test_no_detections_or_tracks_scores_misses_without_crashing(self):
        for row in self.detections:
            row['detections']=[]
        self.tracks=[]
        self.save()
        result=evaluate(self.run,self.data)
        self.assertEqual(result['detected_vehicle_boxes_at_fixed_threshold'],0)
        self.assertEqual(result['coco_metrics']['AP'],0.0)
        self.assertEqual(result['coco_metrics']['AR100'],0.0)
        metrics=result['tracking_metrics']['none']['OVERALL']
        self.assertEqual(metrics['num_misses'],2)
        self.assertEqual(metrics['num_predictions'],0)
        self.assertEqual(metrics['idf1'],0.0)

    def test_changed_ground_truth_is_bound_even_when_localization_scores_agree(self):
        before=evaluate(self.run,self.data)
        self.labels.write_text(self.labels.read_text().replace(' 7 Car ',' 8 Car '))
        after=evaluate(self.run,self.data)
        first=before['ground_truth_provenance'][0]
        second=after['ground_truth_provenance'][0]
        self.assertNotEqual(first['canonical_gt_rows_sha256'],second['canonical_gt_rows_sha256'])
        self.assertNotEqual(first['annotation_file']['sha256'],second['annotation_file']['sha256'])
        self.assertEqual(before['coco_metrics'],after['coco_metrics'])

    def test_missing_duplicate_or_reordered_detection_frames_are_rejected(self):
        original=deepcopy(self.detections)
        for rows in (original[:1],[original[0],original[0]],list(reversed(original))):
            with self.subTest(frames=[row['frame_id'] for row in rows]):
                self.detections=deepcopy(rows)
                self.save()  # Updated file hash must not conceal invalid frame coverage.
                with self.assertRaisesRegex(ValueError,'uniquely cover'):
                    evaluate(self.run,self.data)

    def test_wrong_recording_dimensions_class_confidence_or_box_is_rejected(self):
        original=deepcopy(self.detections)
        for change in ('recording','dimensions','class','confidence','box'):
            with self.subTest(change=change):
                self.detections=deepcopy(original)
                row=self.detections[0]
                if change=='recording':
                    row['recording_id']='other'
                elif change=='dimensions':
                    row['width']=33
                elif change=='class':
                    row['detections'][0]['class_id']=999
                elif change=='confidence':
                    row['detections'][0]['confidence']=float('nan')
                else:
                    row['detections'][0]['bbox_xyxy']=[1,2,99,12]
                self.save()
                with self.assertRaises(ValueError):
                    evaluate(self.run,self.data)

    def test_tracker_sources_frames_duplicates_and_bounds_are_validated(self):
        original=deepcopy(self.tracks)
        for change in ('source','recording','frame','duplicate','box'):
            with self.subTest(change=change):
                self.tracks=deepcopy(original)
                if change=='source':
                    self.tracks[0]['source']='annotated'
                elif change=='recording':
                    self.tracks[0]['recording_id']='other'
                elif change=='frame':
                    self.tracks[0]['frame_id']=3
                elif change=='duplicate':
                    self.tracks.append(deepcopy(self.tracks[0]))
                else:
                    self.tracks[0]['bbox_xyxy']=[1,2,99,12]
                self.save()
                with self.assertRaises(ValueError):
                    evaluate(self.run,self.data)

    def test_current_evaluation_pixels_must_match_inference_hashes(self):
        self.detections[0]['image_sha256']='0'*64
        self.save()
        with self.assertRaisesRegex(ValueError,'pixels differ'):
            evaluate(self.run,self.data)

    def test_unchanged_inference_fingerprints_are_required(self):
        Path(self.info['detections_path']).write_text('')
        with self.assertRaisesRegex(ValueError,'changed after inference'):
            evaluate(self.run,self.data)


if __name__=='__main__':
    unittest.main()
