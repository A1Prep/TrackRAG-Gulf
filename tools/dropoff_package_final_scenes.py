"""Seal genuine Task 1 canonical input scenes and their source receipt. There is no inference or scoring step."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile

from evaluation.grounding import load_scenes
from tools.dropoff_final_handoff_checks import rows
from tools.dropoff_final_input_inventory import ROOT, TASK1, TASK2, TEST_ARCHIVE_SHA, git, require, write
from tools.dropoff_pack_matched import encoded, json_archive
from trackrag.dropoff.data import file_sha256
from trackrag.dropoff.scenes import validate_scene


def package(bank, native, archive, summary_path, sparse_archive=None):
    require(not archive.exists() and not summary_path.exists(), 'Choose fresh canonical archive and summary filenames')
    status = json.loads((native / 'status.json').read_bytes())
    require(status['state'] == 'complete' and status['no_models_called'] and not status['label_files_opened'],
            'All canonical source exports must finish before publication')
    require(status['full_archive_sha_verified'] and status['archive_sha256'] == TEST_ARCHIVE_SHA,
            'Official compressed source SHA has not been verified')
    require(status['source_input_manifest_sha256'] == file_sha256(bank / 'receipt.json'), 'Source inventory differs')
    configuration = json.loads((native / 'run-configuration.json').read_bytes())
    require(configuration['task1_revision'] == TASK1 and not configuration['model_inference'] and
            all(configuration['unmodified_pinned_task1_source_lf_equivalence'].values()), 'Unverified canonical exporter version')
    receipt = json.loads((bank / 'receipt.json').read_bytes())
    coverage = json.loads((bank / 'coverage.json').read_bytes())
    sparse_archive = sparse_archive or bank.parent / 'passenger-44-inputs-only-20261122.zip'
    with ZipFile(sparse_archive) as sparse:
        require(sparse.testzip() is None, 'Saved sparse input archive is unreadable')
        for filename in (*receipt['files'], 'coverage.json', 'receipt.json'):
            require(sparse.read(filename) == (bank / filename).read_bytes(), 'Sparse archive and canonical input sources differ')
    for filename, expected in receipt['files'].items():
        require(file_sha256(bank / filename) == expected, 'Source inventory member changed')
    inputs = {r['expected_task1_scene_sha256']: r for r in rows(bank / 'task1-scene-ingredients.jsonl')}
    scene_dir = native / 'scenes'
    scenes = load_scenes(scene_dir)
    index = json.loads((scene_dir / 'index.json').read_bytes())
    require(set(inputs) == set(scenes) and len(inputs) == len(index) == 44, 'Canonical and input scene sets differ')
    require(file_sha256(scene_dir / 'index.json') == status['scene_index_sha256'], 'Canonical index digest changed')
    archived = {'scenes/index.json': (scene_dir / 'index.json').read_bytes(),
                'run-configuration.json': (native / 'run-configuration.json').read_bytes()}
    frames = json.loads(git(TASK1, 'experiments/vision/emt_frame_sha256_manifest.json'))
    detectors = json.loads(git(TASK1, 'experiments/vision/shared_prediction_runs.json'))
    for record in index:
        path = scene_dir / record['file']
        require(path.resolve().parent == scene_dir.resolve(), 'Scene is outside the declared folder')
        if record['scene_sha256'] not in scenes:
            raise ValueError('The canonical index names an unavailable scene')
        scene = json.loads(path.read_bytes())
        digest = scene['scene_sha256']
        require(digest == record['scene_sha256'] and scene == scenes[digest], 'The scene differs from its index')
        require(scene['model_input'] == inputs[digest]['model_input'] and scene['provenance'] == inputs[digest]['provenance'],
                'Saved predictions or original frame binding differ from the sealed input bank')
        validate_scene(scene, frame_manifest=frames, run_manifest=detectors)
        archived['scenes/' + path.name] = path.read_bytes()

    groups = status['complete_groups']
    require(len(groups) == 5 and sum(g['canonical_scene_count'] for g in groups.values()) == 44,
            'Incomplete recording validation')
    require(all(g['full_data_root_and_shared_detector_validation'] and g['scratch_originals_removed'] for g in groups.values()),
            'The real input prefixes were not finished')
    prefixes = sum(g['prefix_files_verified_individually_after_writing'] for g in groups.values())
    require(prefixes == coverage['physical_prefix']['full_source_frame_ids_needed'], 'Original prefix count differs')
    kinds = {s['context']['vision_profile'] for s in scenes.values()}
    frame_manifests = {s['context']['frame_manifest_sha256'] for s in scenes.values()}
    run_manifests = {s['context']['run_manifest_sha256'] for s in scenes.values()}
    require(len(kinds) == len(frame_manifests) == len(run_manifests) == 1, 'Canonical source conditions changed')
    prior = coverage['prior_input_comparisons']
    summary = {
        'kind': 'original_source_verified_input_only_publication', 'task1_commit': TASK1, 'task2_commit': TASK2,
        'original_source': {'url': status['source'], 'compressed_bytes': status['declared_archive_bytes'],
                            'compressed_sha256_verified': TEST_ARCHIVE_SHA},
        'source_asset_verification': {'recordings': len(groups), 'required_prefix_jpegs_verified_twice': prefixes,
                                      'verified_jpeg_bytes': sum(g['authentic_image_bytes'] for g in groups.values()),
                                      'canonical_scenes': len(scenes), 'index_sha256': status['scene_index_sha256'],
                                      'unmodified_evaluator_reloaded': len(scenes), 'human_label_files_opened': False},
        'saved_detector_condition': {'profile': next(iter(kinds)), 'profile_sha256': file_sha256(ROOT / 'configs/vision/finetuned.json'),
                                     'frame_manifest_sha256': next(iter(frame_manifests)), 'run_manifest_sha256': next(iter(run_manifests)),
                                     'input_records_sha256': receipt['files']['requests-with-saved-tracks.jsonl'],
                                     'observations': coverage['input_checks']['saved_detector_observations']},
        'reviewed_condition': {'original_input_records_sha256': receipt['files']['requests-as-human-reviewed.jsonl'],
                               'rows_with_zero_tracks': coverage['input_checks']['reviewed_zero_observations'],
                               'rows_changed_by_rebinding_saved_detections': coverage['input_checks']['different_reviewed_detector_context']},
        'protocol_state': {'pending_nonconfirmed_cases_excluded': receipt['pending_excluded'], 'final_model_calls': 0, 'formal_score': False,
                           'reference_mask_paths_on_reviewed_source_records': coverage['input_checks']['source_mask_nonnull'],
                           'remaining': coverage['release_gate']},
        'prior_input_overlap': {'known_comparison_banks': len(prior),
                                'banks_with_identifier_or_frame_overlap': [k for k, v in prior.items() if v['official_identifier_leakage'] or any(v['shared_unique'][field] for field in ('request_ids', 'recording_ids', 'images_sha256'))],
                                'reused_normalized_wording_in_new_human_banks': {k: v['final_rows_with_exact_other_wording'] for k, v in prior.items() if k.startswith('new_human')},
                                'scope': coverage['case_history']},
        'sparse_input_archive_sha256': file_sha256(sparse_archive),
        'audit_sha256': file_sha256(bank / 'coverage.json')}
    archived['source-verification.json'] = encoded(summary)
    archived['physical-validation.json'] = encoded({'started_utc': status['started_utc'], 'completed_utc': status['ending_utc'],
                                                   'per_recording': groups, 'full_compressed_source_sha_verified': True})
    digest = json_archive(archive, archived)
    summary['canonical_archive'] = {'filename': archive.name, 'sha256': digest, 'bytes': archive.stat().st_size}
    write(summary_path, summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--sparse-archive', type=Path, help='The sealed earlier input-only request and coverage ZIP')
    parser.add_argument('--summary', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.bank, args.native, args.archive, args.summary, args.sparse_archive)['canonical_archive'], indent=2))
