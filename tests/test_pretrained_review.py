from tools.pretrained_candidates import build_index
from tools.pretrained_review import select
from test_pretrained_candidates import Recording,observation


def test_real_adapter_selection_does_not_use_question_ids_or_future_rows():
    rec=Recording()
    q={'recording_id':'r','question_id':'q','query_frame':10,'question':'Which car moves right?'}
    prefix=[observation(n) for n in range(1,11)]
    inputs=[build_index(rec,rows) for rows in (prefix,prefix+[observation(20,2,'bus')])]
    for variant in ('published-task4','expanded-tfidf-temporal','expanded-tfidf-static'):
        a=select(q,inputs[0],rec,variant,'reserved_current_scene')
        b=select(dict(q,question_id='renamed'),inputs[1],rec,variant,'reserved_current_scene')
        assert a==b
        assert all(row['end_frame']<=10 for row in a)


def test_dense_adapter_consumes_public_query_and_same_candidate_descriptions():
    # Spy verifies the model boundary; it is not a pretrained-quality test.
    class Spy:
        def rank(self,query,text):
            assert 'frame 10' in query and 'Which car' in query
            assert all('q-private-id' not in value for value in text)
            return list(range(len(text)))
    rec=Recording()
    q={'recording_id':'r','question_id':'q-private-id','query_frame':10,'question':'Which car moves right?'}
    result=select(q,build_index(rec,[observation(n) for n in range(1,11)]),rec,
                  'expanded-qwen8b-temporal','reserved_current_scene',Spy())
    assert result[-1]['track_id'] is None and result[-1]['start_frame']==10
