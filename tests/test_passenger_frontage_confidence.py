import base64
from hashlib import sha256
from io import BytesIO
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from tools import passenger_frontage_confidence as gate
from trackrag.dropoff import parking_evidence


class FrontageConfidenceTest(unittest.TestCase):
    def setUp(self):
        self.key = Ed25519PrivateKey.generate()
        public = self.key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
        patcher = patch.object(parking_evidence, 'PUBLIC_KEY', public)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.request = {'text': 'by the shop', 'scene': {'tracks': [], 'provenance': {'image_sha256': 'a'*64},
                                                         'frame': {'frame_id': 1, 'width': 100, 'height': 80}}}
        self.point = {'words': 'by the shop', 'image_sha256': 'a'*64, 'revision': gate.POINT_REVISION,
                      'point_1000': [500, 500]}
        self.maps = {'words': 'by the shop', 'source': 'a'*64, 'revision': gate.SURFACE_REVISION,
                     'queries': np.array([*gate.QUERIES, 'shop']), 'logits': np.zeros((8,96,144))}

    def evidence(self, pixel, reply='BAY', source=None):
        body = {'words': 'by the shop', 'image_sha256': source or self.point['image_sha256'],
                'model': 'trained_qwen3_vl_8b', 'recipe': 'frontage_point_crop_check_v1',
                'point_1000': self.point['point_1000'],
                'calls': [{'stage': 'scene', 'response': 'OPEN'}, {'stage': 'surface', 'response': reply}]}
        if pixel is not None:
            data = BytesIO()
            Image.new('L', (100,80), color=pixel).save(data, format='PNG')
            body['class_map'] = {'sha256': sha256(data.getvalue()).hexdigest(),
                                 'png': base64.b64encode(data.getvalue()).decode()}
        packet = {'body': body, 'signature': self.key.sign(parking_evidence.signed_bytes(body)).hex()}
        return gate.evidence(self.request,self.point,self.maps,packet)

    def test_unknown_and_uncorroborated_ground(self):
        self.assertEqual(self.evidence(None)['veto'], 'no_surface_map')
        self.assertEqual(self.evidence(5)['veto'], 'non_drivable_surface')
        self.assertEqual(self.evidence(13,'LANE')['veto'], 'uncorroborated_road')

    def test_current_box_veto_and_stale_box(self):
        self.request['scene']['tracks'] = [{'frame_id': 0, 'bbox_xyxy': [0,0,100,80]}]
        self.assertIsNone(self.evidence(10)['veto'])
        self.request['scene']['tracks'][0]['frame_id'] = 1
        self.assertEqual(self.evidence(10)['veto'], 'road_user_box')

    def test_authenticated_other_image_rejected(self):
        with self.assertRaisesRegex(ValueError,'different evidence'):
            self.evidence(10,source='b'*64)

    def test_candidate_score_and_veto_precedence(self):
        row = self.evidence(13)
        self.assertEqual(row['features'][-2:], [0.,1.])
        model = {'features': list(gate.FEATURES),'mean': [0.]*8,'scale': [1.]*8,
                 'coefficients': [0.]*8,'intercept': 0.,'threshold': 0.5}
        self.assertTrue(gate.predict(row,model)['accept'])
        decision = gate.predict(self.evidence(None),model)
        self.assertFalse(decision['accept'])
        self.assertIsNone(decision['score'])
