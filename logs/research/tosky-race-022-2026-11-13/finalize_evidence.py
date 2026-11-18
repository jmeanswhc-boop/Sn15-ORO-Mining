"""Rebuild the research report and verify its evidence inventory offline."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
AGENTS = ('chidori', 'tosky', 'pooh')


def main():
    race = json.loads((ROOT/'race-analysis.json').read_text())
    qualifier = json.loads((ROOT/'analysis.json').read_text())
    assert race['complete'] and race['coverage']['available_feedback'] == 540
    assert not race['coverage']['inconsistencies']
    feedback_count = 0
    for folder, phase, expected in [('race-feedback/'+agent, 'RACE', 180) for agent in AGENTS] + [
        ('qualifying-feedback', 'QUALIFYING', 60),
        ('chidori-qualifying-feedback', 'QUALIFYING', 60),
        ('pooh-qualifying-feedback', 'QUALIFYING', 60),
    ]:
        episodes = set()
        for path in (ROOT/folder).glob('*.json'):
            wrapper = json.loads(path.read_text())
            data = wrapper['data']
            assert not wrapper.get('stale') and not wrapper.get('error'), path
            assert data['diagnostics_state'] == 'available' and data['phase'] == phase, path
            assert data['passed'] == bool(data['paid_reward']), path
            assert data['episode_result_id'] not in episodes, path
            episodes.add(data['episode_result_id'])
        assert len(episodes) == expected, (folder, len(episodes))
        feedback_count += len(episodes)
    snapshots = json.loads((ROOT/'source-snapshots/provenance.json').read_text())
    for source in snapshots:
        for path in (ROOT/source['snapshot'], PROJECT/source['original_path']):
            assert hashlib.sha256(path.read_bytes()).hexdigest() == source['sha256'], path
    collection = json.loads((ROOT/'race-collection-state.json').read_text())
    assert collection['complete'] and len(collection['runs']) == 6
    metrics = ['Metric | CHIDORI | Tosky / v4 | Pooh', '--- | --- | --- | ---']
    metrics.append('Qualifying correct / 60 | '+' | '.join(str(qualifier['agents'][a]['qualifying']['correct'])+'/60' for a in AGENTS))
    metrics.append('Race correct / 180 | '+' | '.join(str(race['agents'][a]['correct'])+'/180' for a in AGENTS))
    metrics.append('Race score from binary counts | '+' | '.join(f"{100*race['agents'][a]['correct']/180:.2f}%" for a in AGENTS))
    metrics.append('Mean 90-task cost | '+' | '.join(f"${race['agents'][a]['mean_usd']:.4f}" for a in AGENTS))
    metrics.append('Mean sandbox duration | '+' | '.join(f"{race['agents'][a]['mean_sandbox_seconds']:.1f} s" for a in AGENTS))
    metrics.append('Inference requests, both runs | '+' | '.join(str(race['agents'][a]['total_inferences']) for a in AGENTS))
    metrics.append('Inference errors, both runs | '+' | '.join(str(race['agents'][a]['inference_errors']) for a in AGENTS))
    failures = ['Outcome, out of 180 | CHIDORI | Tosky / v4 | Pooh', '--- | --- | --- | ---']
    for category in ['passed', 'request_not_met', 'process_requirement_missed', 'not_the_best_option',
                     'did_not_finish', 'needs_not_found', 'changes_missed', 'infrastructure']:
        failures.append(category+' | '+' | '.join(str(race['agents'][a]['categories'].get(category, 0)) for a in AGENTS))
    replacements = {
        '@@COVERAGE@@': 'Collection is complete: 540/540 released race feedback records across six included runs, 180/180 qualifying feedback records across six runs, four historical Pooh run indexes and a deliberately selected 12-record historical feedback sample. All 720 current race/qualifier feedback records are available, non-stale public projections. There are zero race identity/reward inconsistencies.',
        '@@METRICS@@': '\n'.join(metrics), '@@FAILURE_TABLE@@': '\n'.join(failures),
        '@@STOPS@@': "Tosky has 20 did_not_finish executions, versus CHIDORI's 2 and Pooh's 1. All 23 end with agent_stopped; none of these six runs reports an episode end caused by the provider inference budget, task timeout or infrastructure. This identifies agent-controlled noncompletion, not its exact internal trigger.",
    }
    report = (ROOT/'recommendation-template.txt').read_text()
    for key, value in replacements.items():
        assert key in report
        report = report.replace(key, value)
    assert '@@' not in report
    for name in ['report.txt', 'recommendation.txt']:
        (ROOT/name).write_text(report)
    verification = {'verified_at': datetime.now(timezone.utc).isoformat(),
                    'race_feedback': 540, 'qualifying_feedback': 180, 'feedback_total': feedback_count,
                    'race_identity_reward_mismatches': 0, 'source_hashes_preserved': snapshots,
                    'race_collection_complete': True, 'agent_modified': False,
                    'paid_inference_or_evaluation_run': False,
                    'scope': 'Evidence integrity and static analysis; not backend agent acceptance or a measured new score.'}
    (ROOT/'verification.json').write_text(json.dumps(verification, indent=2)+'\n')
    files = []
    for path in sorted(ROOT.rglob('*')):
        if path.is_file() and path.name != 'manifest.json':
            data = path.read_bytes()
            files.append({'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)})
    (ROOT/'manifest.json').write_text(json.dumps({'created_at': verification['verified_at'],
        'source_base': 'https://api.oroagents.com/v1/public/', 'uses_paced_local_api_client': True,
        'files': files}, indent=2)+'\n')
    print(json.dumps({'verified_feedback': feedback_count, 'source_hashes_unchanged': len(snapshots),
                      'files': len(files), 'bytes': sum(f['bytes'] for f in files),
                      'report_words': len(report.split())}, indent=2))


if __name__ == '__main__':
    main()
