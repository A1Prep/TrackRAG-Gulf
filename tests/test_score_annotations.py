"""Contributor key adaptation must not masquerade as complete QA scoring."""
import json
import tempfile
import unittest
from pathlib import Path

from evaluation.prepare_annotations import prepare_annotations
from evaluation.score import read_jsonl,score_results


class AnnotationPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.questions=self.root/'questions.jsonl'
        self.key=self.root/'key.jsonl'
        self.q={'question_id':'q','recording_id':'rec','query_frame':20,'question':'Which tracks appear in frames 10 and 20?'}
        self.k={'question_id':'q','expected_answer':'Tracks 1 and 2.','answerable':True,'acceptable_track_ids':[1,2],
                'acceptable_intervals':[{'recording_id':'rec','start_frame':10,'end_frame':20}]}

    def write(self,keys=None):
        self.questions.write_text(json.dumps(self.q)+'\n')
        self.key.write_text(''.join(json.dumps(row)+'\n' for row in (keys or [self.k])))

    def test_schema_join_is_scorable_but_qa_verification_stays_pending(self):
        self.write()
        out=self.root/'packet'
        metadata=prepare_annotations(self.questions,self.key,out)
        gold=read_jsonl(out/'retrieval-key.jsonl')
        self.assertEqual(gold[0]['recording_id'],'rec')
        self.assertEqual(gold[0]['query_frame'],20)
        self.assertEqual(len(gold[0]['acceptable_evidence']),2)
        score_results([],gold)
        qa=read_jsonl(out/'qa-review.jsonl')[0]
        self.assertEqual(qa['explicit_requested_frame_ids'],[10,20])
        self.assertEqual(qa['independent_review_status'],'pending')
        self.assertFalse(metadata['human_validated'])
        self.assertIn('not complete answer correctness',metadata['retrieval_metric_scope'])
        # Input source bytes remain unchanged and separate from inference.
        self.assertEqual(read_jsonl(self.key),[self.k])

    def test_future_intervals_duplicate_keys_and_ambiguous_pairing_fail(self):
        for bad in ('future','duplicate','pairing'):
            with self.subTest(case=bad):
                self.k['acceptable_intervals']=[{'recording_id':'rec','start_frame':10,'end_frame':21 if bad=='future' else 20}]
                if bad=='pairing':
                    self.k['acceptable_intervals'].append({'recording_id':'rec','start_frame':1,'end_frame':2})
                self.write([self.k,self.k] if bad=='duplicate' else [self.k])
                with self.assertRaises(ValueError):
                    prepare_annotations(self.questions,self.key,self.root/bad)

    def test_unanswerable_key_is_separate_from_missing_model_response(self):
        self.k.update(answerable=False,acceptable_track_ids=[],acceptable_intervals=[],expected_answer='The footage cannot establish that.')
        self.write()
        out=self.root/'unanswerable'
        metadata=prepare_annotations(self.questions,self.key,out)
        self.assertEqual(metadata['unanswerable_count'],1)
        self.assertEqual(read_jsonl(out/'retrieval-key.jsonl')[0]['acceptable_evidence'],[])


if __name__=='__main__':
    unittest.main()
