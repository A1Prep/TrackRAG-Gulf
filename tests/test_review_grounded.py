"""Offline grounding regressions: real evidence files, replaced model replies."""
import json
import os
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from evaluation.run_audit import audit_run
from evaluation.score import read_jsonl
from tools import grounded_trial_adapters as grounded
from trackrag.review import review_recording


class Frames:
    recording_id='a'
    def __init__(self,root):
        self.frames={}
        for number in range(1,7):
            path=root/f'{number}.png'
            Image.new('RGB',(32,24),(number,40,80)).save(path)
            self.frames[number]={'recording_id':'a','frame_id':number,'width':32,'height':24,
                                 'timestamp_s':(number-1)/10,'image_path':str(path)}
    def iter_frames(self):
        return iter(self.frames.values())
    def get_frame(self,number):
        return self.frames[number]


class GroundedTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.recording=Frames(self.root)
        self.rows=[{'recording_id':'a','source':'annotated','track_id':7,'frame_id':number,
                    'class_name':'Motorbike','bbox_xyxy':[4,5,20,18]} for number in range(1,7)]
        self.question={'question_id':'q','recording_id':'a','query_frame':5,'question':'Does it grow between frames2 and5?'}
        self.env=patch.dict(os.environ,{'TRACKRAG_TRIAL_CACHE':str(self.root/'cache')})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.calls=[]

    def reply(self,parts,schema,role,receipt_path=None,metadata=None,**kwargs):
        self.calls.append((role,parts,kwargs))
        response={'selected':['T7-F1-F5'],'requested_frame_ids':[2,5]} if role=='evidence selection' else {
            'answer_status':'answered','answer':'The annotated motorcycle grows between frames2 and5.','cited_evidence_ids':['E1']}
        kwargs['validate'](response)
        if receipt_path:
            Path(receipt_path).write_text(json.dumps({'model':'test-model','role':role,'cached':False,
                'request_status':'completed','response':response,**(metadata or {})}),encoding='utf-8')
        return response

    def run_case(self,question=None):
        directory=self.root/'run'
        question=question or self.question
        with patch.object(grounded,'request_json',side_effect=self.reply):
            review_recording(self.recording,[question],source='annotated',load_tracks=lambda rec:self.rows,
                             build_index=grounded.build_index,rank_evidence=grounded.rank_evidence,
                             answer_question=grounded.answer_question,export_retrieval=grounded.export_retrieval,
                             output_dir=directory)
        return directory,read_jsonl(directory/'results.jsonl')[0]

    def test_endpoint_refinement_query_scene_trace_and_fresh_answer(self):
        directory,row=self.run_case()
        self.assertEqual(row['answer']['answer_status'],'answered')
        self.assertEqual([(e['interval']['track_id'],e['interval']['start_frame'],e['interval']['end_frame']) for e in row['evidence']],[(7,2,5),(None,5,5)])
        self.assertTrue({2,5}.issubset({image['frame_id'] for image in row['evidence'][0]['images']}))
        self.assertEqual(self.calls[-1][2]['use_cache'],False)
        self.assertEqual(self.calls[-1][2]['max_output_tokens'],2048)
        self.assertIn('Motorbike',self.calls[-1][1][0]['text'])
        audited=audit_run(directory,{'q':self.question})
        self.assertTrue(any(file['role']=='selector_input' for file in audited['files']))

    def test_numeric_track_reference_uses_supplied_identity_without_selector(self):
        question=dict(self.question,question='What is annotated as track7?')
        directory,row=self.run_case(question)
        self.assertEqual(row['evidence'][0]['interval']['track_id'],7)
        self.assertEqual([role for role,_,_ in self.calls],['grounded answer'])
        audit_run(directory,{'q':question})

    def test_future_rows_cannot_change_stable_candidate_tokens_or_gallery_metadata(self):
        index=grounded.build_index(self.recording,self.rows)
        before=grounded.candidates_at_query(self.question,index)
        self.assertTrue(all(c['candidate_id'].startswith(('T7-F','scene-F')) for c in before))
        index['tracks'][999]=[dict(self.rows[-1],track_id=999,class_name='FutureSecret')]
        self.assertEqual(before,grounded.candidates_at_query(self.question,index))
        _,sent=grounded.identity_views(self.question,index)
        self.assertEqual([v['track_id'] for gallery in sent for v in gallery['identities']],[7])
        self.assertTrue(all(v['frame_id']<=5 for gallery in sent for v in gallery['identities']))

    def test_future_or_escaping_selection_trace_is_rejected(self):
        directory,_=self.run_case()
        path=next(directory.glob('hosted/*/selection-receipt.json'))
        original=json.loads(path.read_text())
        for kind in ('future','identity','path','interval'):
            value=deepcopy(original)
            if kind=='future': value['selection_images'][0]['frame_id']=6
            elif kind=='identity': value['identity_images'][0]['identities'][0]['track_id']=999
            elif kind=='path': value['selection_images'][0]['image']='../../outside.png'
            else: value['returned_intervals'][0]['track_id']=8
            path.write_text(json.dumps(value))
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                audit_run(directory,{'q':self.question})

    def test_unprovided_or_future_frame_claim_fails_validation(self):
        directory,row=self.run_case()
        # Exercise the actual answer validator, not a mirrored implementation.
        def invalid(parts,schema,role,receipt_path=None,metadata=None,**kwargs):
            response={'answer_status':'answered','answer':'It grows through frame6.','cited_evidence_ids':['E1']}
            kwargs['validate'](response)
        with patch.object(grounded,'request_json',side_effect=invalid),self.assertRaisesRegex(ValueError,'exact frames'):
            grounded.answer_question(self.question,row['evidence'],directory)

    def test_wrong_cited_track_id_is_rejected(self):
        directory,row=self.run_case()
        def invalid(parts,schema,role,receipt_path=None,metadata=None,**kwargs):
            kwargs['validate']({'answer_status':'answered','answer':'Track99 grows.','cited_evidence_ids':['E1']})
        with patch.object(grounded,'request_json',side_effect=invalid),self.assertRaisesRegex(ValueError,'track ID'):
            grounded.answer_question(self.question,row['evidence'],directory)

    def test_annotation_identity_is_not_silently_mapped_to_predictions(self):
        with self.assertRaisesRegex(ValueError,'annotated'):
            grounded.build_index(self.recording,[dict(row,source='predicted') for row in self.rows])

    def test_frame_references_do_not_confuse_track_ids_or_units(self):
        self.assertEqual(grounded.explicit_frames('Track7 at frames117 and150, 80 kilometres per hour'),[117,150])
        self.assertEqual(grounded.explicit_frames('frame2 to frame5'),[2,5])
        self.assertEqual(grounded.explicit_frames('Compare track7 in frames2,4.'),[2,4])
        self.assertEqual(grounded.explicit_frames('Compare frames2,4, and5'),[2,4,5])
        self.assertEqual(grounded.explicit_frames('track6 in the opening frames'),[])

    def test_temporal_endpoints_from_different_identities_cannot_be_combined(self):
        directory,row=self.run_case()
        evidence=deepcopy(row['evidence'])
        # This deliberately gives each identity only one endpoint.
        evidence[0]['images']=[evidence[0]['images'][0]]
        evidence[0]['interval'].update(start_frame=2,end_frame=2)
        evidence[1]['interval'].update(track_id=8)
        evidence[1]['images'][0]['bbox_xyxy']=[4,5,20,18]
        def invalid(parts,schema,role,receipt_path=None,metadata=None,**kwargs):
            kwargs['validate']({'answer_status':'answered','answer':'Track7 grows between frames2 and5.','cited_evidence_ids':['E1','E2']})
        with patch.object(grounded,'request_json',side_effect=invalid),self.assertRaisesRegex(ValueError,'same stated track'):
            grounded.answer_question(self.question,evidence,directory)

    def test_missing_comma_list_endpoint_cannot_be_silently_ignored(self):
        directory,row=self.run_case()
        evidence=deepcopy(row['evidence'])
        evidence[0]['images']=[image for image in evidence[0]['images'] if image['frame_id']!=4]
        question=dict(self.question,question='Compare track7 in frames2,4.')
        def invalid(parts,schema,role,receipt_path=None,metadata=None,**kwargs):
            kwargs['validate']({'answer_status':'answered','answer':'Track7 grows.','cited_evidence_ids':['E1']})
        with patch.object(grounded,'request_json',side_effect=invalid),self.assertRaisesRegex(ValueError,'exact frames'):
            grounded.answer_question(question,evidence,directory)

    def test_scene_backed_absence_can_use_historical_track_and_current_scene(self):
        directory,row=self.run_case()
        evidence=deepcopy(row['evidence'])
        evidence[0]['images']=[evidence[0]['images'][0]]
        evidence[0]['interval'].update(start_frame=2,end_frame=2)
        question=dict(self.question,question='Is track7 still visible in frame5?')
        expected={'answer_status':'answered','answer':'Track7 was visible in frame2 but is not visible in frame5.','cited_evidence_ids':['E1','E2']}
        def reply(parts,schema,role,receipt_path=None,metadata=None,**kwargs):
            kwargs['validate'](expected)
            return expected
        with patch.object(grounded,'request_json',side_effect=reply):
            self.assertEqual(grounded.answer_question(question,evidence,directory),expected)


if __name__=='__main__':
    unittest.main()
