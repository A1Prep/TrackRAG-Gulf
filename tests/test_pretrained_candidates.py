import tempfile
import unittest
from pathlib import Path

from tools.pretrained_candidates import build_index,candidate_catalog,describe,tfidf_scores,ranked_intervals,reserve_current_scene
from tools.local_embeddings import LocalEmbeddings


class Recording:
    recording_id='r'
    def get_frame(self,n):
        if not 1<=n<=20:
            raise KeyError(n)
        return {'recording_id':'r','frame_id':n,'width':100,'height':80,'image_path':'unused','timestamp_s':n/10}
    def iter_frames(self):
        return iter(self.get_frame(n) for n in range(1,21))


def observation(n,track=1,label='car'):
    return {'recording_id':'r','frame_id':n,'track_id':track,'class_name':label,'bbox_xyxy':[n,10,n+20,30],'source':'predicted'}


class CandidateTests(unittest.TestCase):
    def test_future_rows_do_not_change_candidates_text_or_rank(self):
        recording=Recording()
        question={'recording_id':'r','question_id':'arbitrary','query_frame':10,'question':'Compare the car at frames 1 and 10.'}
        prefix=[observation(n) for n in range(1,11)]
        full=prefix+[observation(n) for n in range(11,21)]+[observation(20,2,'bus')]
        pools=[candidate_catalog(question,build_index(recording,rows)) for rows in (prefix,full)]
        self.assertEqual(pools[0],pools[1])
        self.assertTrue(all(c['end_frame']<=10 for c in pools[0]))
        descriptions=[[describe(c,recording) for c in pool] for pool in pools]
        self.assertEqual(descriptions[0],descriptions[1])
        self.assertEqual(ranked_intervals(pools[0],tfidf_scores(question['question'],descriptions[0])),ranked_intervals(pools[1],tfidf_scores(question['question'],descriptions[1])))
        self.assertTrue(any(c['track_id']==1 and (c['start_frame'],c['end_frame'])==(1,10) for c in pools[0]))
        self.assertTrue(any(c['track_id'] is None and c['start_frame']==10 for c in pools[0]))

    def test_future_only_source_has_no_scene_and_id_does_not_affect_text(self):
        recording=Recording()
        q={'recording_id':'r','question_id':'a','query_frame':10,'question':'car'}
        self.assertEqual(candidate_catalog(q,build_index(recording,[observation(20)])),[])
        a=candidate_catalog(q,build_index(recording,[observation(1,1)]))
        b=candidate_catalog(dict(q,question_id='different'),build_index(recording,[observation(1,99)]))
        self.assertEqual([describe(c,recording) for c in a],[describe(c,recording) for c in b])
        self.assertFalse(any('99' in describe(c,recording) for c in b))

    def test_common_scene_policy_preserves_two_ranked_items(self):
        recording=Recording()
        q={'recording_id':'r','question_id':'q','query_frame':10,'question':'car'}
        pool=candidate_catalog(q,build_index(recording,[observation(n) for n in range(1,11)]))
        raw=ranked_intervals(pool,list(range(len(pool))))
        reserved=reserve_current_scene(raw,pool)
        self.assertEqual([r['rank'] for r in reserved],list(range(1,len(reserved)+1)))
        self.assertEqual(reserved[-1]['track_id'],None)
        self.assertEqual(reserved[-1]['start_frame'],10)
        with self.assertRaises(ValueError):
            ranked_intervals(pool,[float('nan')]*len(pool))

    def test_cache_identity_includes_runtime(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'cache.sqlite'
            a=LocalEmbeddings('http://127.0.0.1:1','a'*64,path,runtime_id='r1')
            # Insert an actual-format cache row to test namespace separation,
            # without treating this fixture as model inference.
            import array,hashlib
            key=hashlib.sha256(('a'*64+'|r1|last-pooling|hello').encode()).hexdigest()
            a.cache.execute('INSERT INTO vectors VALUES(?,?,?,?,?)',(key,'hello',array.array('f',[1,0]).tobytes(),2,0))
            a.cache.commit()
            self.assertEqual(a.embed(['hello']),[[1.0,0.0]])
            b=LocalEmbeddings('http://127.0.0.1:1','a'*64,path,runtime_id='r2')
            with self.assertRaises(OSError):
                b.embed(['hello'])
            a.close();b.close()


if __name__=='__main__':
    unittest.main()
