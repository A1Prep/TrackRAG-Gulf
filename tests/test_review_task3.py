"""Normal Task 3 and reviewer wiring, synthetic footage and injected replies."""
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from evaluation.run_audit import audit_run
from evaluation.score import read_jsonl
from trackrag.__main__ import main,check_adapters
from trackrag.evidence_time import requested_frame_ids


class Task3IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def make_recording(self,name,frames=3):
        images=self.root/'images/train'/name
        images.mkdir(parents=True)
        labels=self.root/'annotations/train'
        labels.mkdir(parents=True,exist_ok=True)
        lines=[]
        for frame in range(1,frames+1):
            Image.new('RGB',(40,30),(100,150,200)).save(images/f'{frame:05d}.jpg')
            lines.append(f'{frame} 7 Bus 0 0 0 3 4 20 22 0 0 0 0 0 0 0\n')
        (labels/f'{name}.txt').write_text(''.join(lines))
        return [{'question_id':name+f'-q{i}','recording_id':name,'query_frame':frames,
                 'question':'Which bus is visible?'} for i in range(4)]

    def test_normal_answerer_with_explicit_retrieval_override_on_two_recordings(self):
        checks=check_adapters('annotated','gemini',retrieval_adapter='tools.emt_trial_adapters')
        self.assertTrue(checks['ready'],checks)
        self.assertIn('trackrag.answering.answer_question',[row['function'] for row in checks['adapters']])
        for name in ('rec_a','rec_b'):
            questions=self.make_recording(name)
            path=self.root/f'{name}-questions.jsonl'
            path.write_text(''.join(json.dumps(q)+'\n' for q in questions))
            seen=[]
            class OfflineModel:
                model_name='offline-integration-substitute'
                def generate(self,question,evidence):
                    seen.append((question,evidence))
                    if len(seen)==2:
                        return {'answer_status':'insufficient_evidence','answer':'The requested detail is not established.','cited_evidence_ids':[]}
                    if len(seen)==3:
                        raise TimeoutError('Offline simulated timeout')
                    return {'answer_status':'answered','answer':'The bus in E1.','cited_evidence_ids':['E1']}
            output=self.root/f'{name}-run'
            with patch('trackrag.model_client.GeminiAnswerModel',OfflineModel),redirect_stdout(io.StringIO()):
                code=main(['review','--data-root',str(self.root),'--recording',name,'--questions',str(path),
                    '--source','annotated','--answerer','gemini','--retrieval-adapter','tools.emt_trial_adapters','--out',str(output)])
            self.assertEqual(code,0)
            self.assertEqual(len(seen),4)
            results=read_jsonl(output/'results.jsonl')
            self.assertEqual([row['answer']['answer_status'] for row in results],['answered','insufficient_evidence','error','answered'])
            audited=audit_run(output,{q['question_id']:q for q in questions})
            self.assertTrue(any(row['role']=='provider_input' for row in audited['files']))
            for question in questions:
                hosted=output/'hosted'/sha256(question['question_id'].encode()).hexdigest()
                receipt=json.loads((hosted/'receipt.json').read_text())
                for image in receipt['sent']:
                    self.assertEqual(sha256((hosted/image['image']).read_bytes()).hexdigest(),image['sha256'])
            metadata=json.loads((output/'run.json').read_text())['configuration']
            self.assertEqual(metadata['answer_adapter'],'trackrag.answering')
            self.assertEqual(metadata['retrieval_adapter'],'tools.emt_trial_adapters')

    def test_explicit_frame_parser_preserves_all_four_times_and_ignores_ids_units(self):
        self.assertEqual(requested_frame_ids('Does track 33 appear in frames 205, 230, 260 and 294?'),[205,230,260,294])
        self.assertEqual(requested_frame_ids('track 35 at frame 205 and 80 kilometres per hour'),[205])

    def test_four_requested_observations_are_actually_exported(self):
        questions=self.make_recording('four',frames=7)[:1]
        questions[0]['question']='Does the bus appear in frames 1, 2, 6 and 7?'
        from trackrag.emt import open_recording,load_annotated_tracks
        from trackrag.review import review_recording
        from tools.emt_trial_adapters import build_index,rank_evidence
        output=self.root/'four-run'
        def rank(question,index,k=3):
            return [{'recording_id':'four','source':'annotated','track_id':7,'start_frame':1,'end_frame':7,'rank':1,'score':1.0}]
        review_recording(open_recording(self.root,'four'),questions,source='annotated',
            load_tracks=load_annotated_tracks,build_index=build_index,rank_evidence=rank,output_dir=output)
        row=read_jsonl(output/'results.jsonl')[0]
        self.assertTrue({1,2,6,7}.issubset({image['frame_id'] for image in row['evidence'][0]['images']}))
        audit_run(output,{questions[0]['question_id']:questions[0]})


if __name__=='__main__':
    unittest.main()
