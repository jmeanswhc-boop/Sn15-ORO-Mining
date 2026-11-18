"""Offline contract and archive regressions; no public API or agent execution."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch, MagicMock
import zipfile

from registry import Registry, Worker, unpack_source, public_qualifier
import server

A = '12345678-1234-1234-1234-123456789abc'
B = '22345678-1234-1234-1234-123456789abc'
HOTKEY = '5' + 'A' * 47
SOURCE = b'def agent_main(problem_data):\n    return {}\n'


class API:
    def __init__(self, paths):
        self.paths = paths
        self.calls = []

    def get(self, path):
        self.calls.append(path)
        value = self.paths[path]
        if isinstance(value, Exception):
            raise value
        return {'data': value, 'error': None, 'stale': False}

    def artifact_url(self, ident):
        self.calls.append('artifact:' + ident)
        return {'download_url': 'https://storage.example/source'}


class Response(io.BytesIO):
    headers = {}


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = Registry(self.temp.name, HOTKEY)
        self.store.set_meta('suites', [{'suite_id': 160, 'is_active': True, 'execution_kind': 'envpack'}])
        self.status = {'agent_version_id': A, 'agent_name': 'Tosky', 'version_number': 1,
                       'suite_id': 160, 'miner_hotkey': HOTKEY, 'final_score': 0,
                       'state': 'ELIGIBLE', 'code_available_at': '2020-01-01T00:00:00'}
        self.api = API({'agent-versions/' + A + '/status': self.status})
        self.worker = Worker(self.store, self.api)

    def test_identity_uses_fixed_hotkey_and_survives_restart(self):
        self.store.upsert(self.status)
        self.store.upsert(dict(self.status, agent_version_id=B, miner_hotkey='other'))
        restarted = Registry(self.temp.name)
        self.assertEqual(restarted.meta('config')['hotkey'], HOTKEY)
        self.assertEqual([x['agent_version_id'] for x in restarted.snapshot('tosky')['entries']], [A])

    def test_champion_zero_and_unpublished_scores_are_distinct(self):
        self.store.upsert(self.status, champion=True, title={'top_at': '2026-01-01', 'final_score': 0})
        snap = self.store.snapshot()
        self.assertEqual(snap['entries'][0]['final_score'], 0)
        self.assertIsNone(snap['entries'][0].get('race_score'))
        self.assertIsNone(snap['entries'][0].get('eliminated_at'))
        self.assertEqual(snap['summary']['champions'], 1)

    def test_legacy_race_excluded_and_race_winner_not_a_titleholder(self):
        race = {'race_id': B, 'suite_id': 3, 'race_number': 1, 'winner_agent_version_id': A}
        self.store.record_race({'race': race, 'qualifiers': [self.status]})
        self.assertIsNone(self.store.version(A))
        race['suite_id'] = 160
        self.store.record_race({'race': race, 'qualifiers': [dict(self.status, race_score=0, eliminated_at='2026-01-01')]})
        self.assertFalse(self.store.version(A)['champion'])
        self.assertIsNone(json.loads(self.store.version(A)['data']).get('eliminated_at'))
        self.assertEqual(self.store.snapshot('tosky')['entries'][0]['races'][0]['result']['race_score'], 0)

    def test_pagination_checkpoint_and_unfiltered_ranking(self):
        path = 'leaderboard?suite_id=160&score_type=qualifying&limit=100&offset=0'
        self.api.paths[path] = {'entries': [dict(self.status, rank=101)], 'total': 2}
        self.worker.inventory_page()
        self.assertEqual(self.store.meta('backfill')['leaderboards']['160:qualifying']['offset'], 1)
        self.assertEqual(self.store.snapshot('tosky')['entries'][0]['qualifying_rank'], 101)
        self.api.paths[path.replace('offset=0', 'offset=1')] = {'entries': [dict(self.status, agent_version_id=B, rank=102)], 'total': 2}
        resumed = Worker(Registry(self.temp.name), self.api)
        resumed.inventory_page()
        scan = self.store.meta('backfill')['leaderboards']['160:qualifying']
        self.assertEqual(scan['passes'], 1)
        self.assertEqual(len(self.store.snapshot('archive')['entries']), 2)

    def test_truncated_pagination_does_not_claim_complete(self):
        self.api.paths['leaderboard?suite_id=160&score_type=qualifying&limit=100&offset=0'] = {'entries': [], 'total': 10}
        with self.assertRaises(RuntimeError):
            self.worker.inventory_page()
        self.assertFalse(self.store.meta('backfill', {}).get('completed_at'))

    def test_future_release_is_scheduled_without_artifact_call(self):
        self.status['code_available_at'] = '2099-01-01T00:00:00'
        self.store.upsert(self.status)
        self.worker.download(self.store.version(A))
        row = self.store.version(A)
        self.assertEqual(row['download_state'], 'waiting_release')
        self.assertGreater(row['next_try'], time.time())
        self.assertFalse(any(x.startswith('artifact:') for x in self.api.calls))

    def test_release_gated_error_is_retryable(self):
        self.store.upsert(self.status)
        error = RuntimeError('gated')
        error.release_gated = True
        with patch.object(self.api, 'artifact_url', side_effect=error):
            self.worker.download(self.store.version(A))
        row = self.store.version(A)
        self.assertEqual(row['download_state'], 'waiting_release')
        self.assertEqual(row['error'], 'ARTIFACT_NOT_RELEASED')

    @patch('registry.urlopen')
    def test_download_hash_restart_and_champion_promotion(self, request):
        request.return_value = Response(SOURCE)
        self.store.upsert(self.status)
        self.worker.download(self.store.version(A))
        path = self.store.archive_file(A)
        self.assertEqual(path.read_bytes(), SOURCE)
        self.assertEqual(self.store.version(A)['sha256'], hashlib.sha256(SOURCE).hexdigest())
        self.assertTrue(str(path).startswith(str(Path(self.temp.name) / 'challengers')))
        self.store.upsert(self.status, champion=True)
        self.store.promote_archives()
        self.store.export_catalog()
        restarted = Registry(self.temp.name)
        self.assertEqual(restarted.archive_file(A).read_bytes(), SOURCE)
        self.assertEqual(restarted.archive_file(A).parent.parent.name, 'champions')
        self.assertTrue(path.exists())
        metadata = json.loads((restarted.archive_file(A).parent / 'metadata.json').read_text())
        self.assertNotIn('storage.example', json.dumps(metadata))
        self.assertEqual(len(json.loads((Path(self.temp.name) / 'champions/registry.json').read_text())['champions']), 1)

    def test_stale_response_cannot_advance_inventory(self):
        self.api.get = lambda path: {'data': {}, 'error': None, 'stale': True}
        with self.assertRaises(RuntimeError):
            self.worker.inventory_page()
        self.assertEqual(self.store.snapshot('archive')['total'], 0)

    def test_existing_archive_requires_matching_hash(self):
        folder = Path(self.temp.name) / 'champions/existing'
        folder.mkdir(parents=True)
        (folder / 'agent.py').write_bytes(SOURCE)
        meta = {'api_detail': self.status, 'artifact_file': 'agent.py', 'archive_sha256': 'wrong'}
        (folder / 'metadata.json').write_text(json.dumps(meta))
        self.store.import_champions()
        self.assertIsNone(self.store.archive_file(A))
        meta['archive_sha256'] = hashlib.sha256(SOURCE).hexdigest()
        (folder / 'metadata.json').write_text(json.dumps(meta))
        self.store.import_champions()
        self.assertEqual(self.store.archive_file(A).read_bytes(), SOURCE)
        self.assertFalse(self.store.version(A)['champion'])

    def test_title_updates_do_not_duplicate_events_or_erase_scores(self):
        title = {'suite_id': 160, 'top_at': '2026-01-01', 'final_score': 0.4, 'is_current_top': True}
        self.store.upsert(self.status, champion=True, title=title)
        self.store.upsert(self.status, champion=True, title=dict(title, is_current_top=False))
        self.store.upsert(self.status, champion=True, title=dict(title, final_score=None))
        titles = self.store.snapshot()['entries'][0]['titles']
        self.assertEqual(len(titles), 1)
        self.assertEqual(titles[0]['final_score'], 0.4)

    def test_release_due_preempts_bulk_pending_sources(self):
        self.store.set_meta('race_snapshots_imported', True)
        self.store.set_meta('turn', 3)
        self.store.upsert(self.status)
        self.store.upsert(dict(self.status, agent_version_id=B))
        self.worker.defer(B, 'waiting_release', time.time() - 1, None)
        with patch.object(self.worker, 'download') as download:
            self.worker.step()
        self.assertEqual(download.call_args[0][0]['id'], B)

    def test_imported_race_snapshots_do_not_replace_live_results(self):
        race = {'race_id': B, 'suite_id': 160, 'race_number': 1}
        self.store.record_race({'race': race, 'qualifiers': [dict(self.status, race_score=0.8)]})
        path = Path(self.temp.name) / 'problems/races/example/race_detail.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({'race': race, 'qualifiers': [dict(self.status, race_score=0.1)]}))
        self.store.import_race_snapshots()
        self.assertEqual(self.store.snapshot('tosky')['entries'][0]['races'][0]['result']['race_score'], 0.8)

    def test_coldkey_has_finalized_block_provenance(self):
        self.worker.chain_url = 'https://chain.example'
        chain = MagicMock()
        chain.get_chain_finalised_head.return_value = '0xfinalized'
        chain.query.return_value.value = 'public-coldkey'
        self.worker.chain = chain
        self.worker.owner_step()
        owner = self.store.meta('owner:' + HOTKEY)
        self.assertEqual(owner['coldkey'], 'public-coldkey')
        self.assertEqual(owner['block_hash'], '0xfinalized')
        chain.query.assert_called_once_with('SubtensorModule', 'Owner', [HOTKEY], block_hash='0xfinalized')

    def test_fast_discovery_never_uses_search_rank_as_global_rank(self):
        self.store.set_meta('imported', True)
        self.store.set_meta('suite_checked', time.time())
        self.api.paths.update({'top': {'suite_id': 160, 'top_agent_version_id': None},
            'top/history': {'entries': [dict(self.status, suite_id=3)]},
            'races/current': {'race': None}, 'evaluations/running': [],
            'races/history?limit=100&offset=0': {'races': [], 'total': 0},
            'evaluations/pending': {'items': []},
            'leaderboard?q=' + HOTKEY + '&limit=100': {'suite': {'suite_id': 160},
                'entries': [dict(self.status, rank=1)]},
            'agent-versions/' + A + '/runs': []})
        self.worker.fast()
        item = self.store.snapshot('tosky')['entries'][0]
        self.assertNotIn('rank', item)
        self.assertNotIn('qualifying_rank', item)
        self.assertFalse(item['champion'])

    def test_fast_poll_discovers_completion_while_priority_queue_is_busy(self):
        self.store.set_meta('imported', True)
        self.store.set_meta('suite_checked', time.time())
        self.api.paths.update({'top': {'suite_id': 160, 'top_agent_version_id': None},
            'top/history': {'entries': []}, 'races/current': {'race': None},
            'evaluations/running': [], 'evaluations/pending': [],
            'leaderboard?q=' + HOTKEY + '&limit=100': {'suite': {'suite_id': 160}, 'entries': []}})
        self.worker.race_archive.scheduling.source=MagicMock(return_value={'id': A})
        self.worker.race_page=MagicMock()
        self.worker.fast()
        self.worker.race_page.assert_called_once_with()

    def test_historical_qualifier_keeps_its_suite_and_does_not_replace_zero(self):
        races = [{'race': {'suite_id': 146, 'race_id': B}, 'result': {'qualifying_score': 0.9}}]
        result = public_qualifier({'suite_id': 160, 'final_score': None}, races)
        self.assertEqual(result['score'], 0.9)
        self.assertEqual(result['suite_id'], 146)
        self.assertEqual(result['source'], 'historical_race_roster')
        self.assertEqual(public_qualifier({'suite_id': 160, 'final_score': 0}, races)['score'], 0)

    def test_race_ranking_uses_documented_sort_and_distinct_scope(self):
        self.store.set_meta('backfill', {'leaderboards': {'160:qualifying': {'offset': 0, 'passes': 1, 'next_scan': time.time()+300}}})
        self.api.paths['leaderboard?suite_id=160&score_type=race&limit=100&offset=0'] = {
            'entries': [dict(self.status, rank=37, race_score=0.4)], 'total': 1}
        self.worker.inventory_page()
        item = self.store.snapshot('tosky')['entries'][0]
        self.assertEqual(item['global_race_rank'], 37)
        self.assertNotIn('overall_rank', item)
        self.assertNotIn('qualifying_rank', item)

    def test_champion_search_matches_name_and_both_keys_before_pagination(self):
        self.store.upsert(dict(self.status, agent_name='ALPHA', miner_hotkey='Hot-A'), champion=True)
        self.store.upsert(dict(self.status, agent_version_id=B, agent_name='Beta', miner_hotkey='Hot-B'), champion=True)
        self.store.set_meta('owner:Hot-B', {'coldkey': 'Cold-B'})
        for query in ['beta', 'hot-b', 'COLD-B']:
            result = self.store.snapshot(query=query, limit=1)
            self.assertEqual(result['total'], 1)
            self.assertEqual(result['entries'][0]['agent_version_id'], B)
        self.assertEqual(self.store.snapshot(query='not present')['total'], 0)
        self.assertEqual(self.store.snapshot(query='   ')['total'], 2)

    def test_champion_score_sorts_match_display_and_zero_precedes_missing(self):
        self.store.upsert(dict(self.status, final_score=0.1), champion=True,
                          title={'suite_id':160, 'top_at':'2026-01-01', 'final_score':0.9})
        self.store.upsert(dict(self.status, agent_version_id=B, final_score=0.8), champion=True,
                          title={'suite_id':160, 'top_at':'2026-02-01', 'final_score':0})
        self.assertEqual(self.store.snapshot(sort_by='overall_score')['entries'][0]['agent_version_id'], A)
        self.assertEqual(self.store.snapshot(sort_by='qualifier_score')['entries'][0]['agent_version_id'], B)
        c='32345678-1234-1234-1234-123456789abc'
        self.store.upsert(dict(self.status, agent_version_id=c, final_score=None), champion=True)
        for order, expected in [('asc',[B,A,c]), ('desc',[A,B,c])]:
            self.assertEqual([i['agent_version_id'] for i in self.store.snapshot(sort_by='overall_score',order=order)['entries']], expected)
        self.store.set_meta('top', {'top_agent_version_id':B,'top_score':1})
        self.assertEqual(self.store.snapshot(sort_by='overall_score')['entries'][0]['champion_score'], 1)

    def test_wallet_sorts_use_full_keys_and_unknowns_stay_last(self):
        self.store.upsert(dict(self.status, miner_hotkey=None), champion=True)
        self.store.upsert(dict(self.status, agent_version_id=B, miner_hotkey='Hot-B'), champion=True)
        self.store.set_meta('owner:Hot-B', {'coldkey':'Cold-B'})
        for field in ['hotkey','coldkey']:
            for order in ['asc','desc']:
                result=self.store.snapshot(sort_by=field,order=order)
                self.assertEqual(result['entries'][0]['agent_version_id'], B)
                self.assertEqual(result['entries'][1]['agent_version_id'], A)

    def test_champion_order_uses_first_title_not_submission_or_latest_title(self):
        self.store.upsert(self.status, champion=True, title={'suite_id':160,'top_at':'2026-01-01'})
        self.store.upsert(self.status, champion=True, title={'suite_id':160,'top_at':'2026-03-01'})
        self.store.upsert(dict(self.status,agent_version_id=B),champion=True,
                          title={'suite_id':160,'top_at':'2026-02-01'})
        first=self.store.snapshot(order='asc',limit=1)
        self.assertEqual(first['total'],2)
        self.assertEqual(first['entries'][0]['agent_version_id'],A)
        self.assertEqual(first['entries'][0]['champion_since'],'2026-01-01')
        self.assertEqual(self.store.snapshot(order='desc',limit=1)['entries'][0]['agent_version_id'],B)
        self.assertEqual(self.store.snapshot(order='asc',offset=1,limit=1)['entries'][0]['agent_version_id'],B)

    def test_champion_filter_parameters_are_bounded(self):
        for params in [{'query':'x'*201},{'sort_by':'submitted_at'},{'order':'random'}]:
            with self.assertRaises(ValueError): self.store.snapshot(**params)

    def test_becoming_champion_checks_code_without_old_release_delay(self):
        self.store.upsert(self.status)
        self.worker.defer(A,'waiting_release',time.time()+3600,None)
        self.store.upsert(self.status,champion=True)
        self.assertEqual(self.store.version(A)['next_try'],0)
        self.worker.defer(A,'waiting_release',time.time()+300,'ARTIFACT_NOT_RELEASED')
        self.store.upsert(self.status,champion=True)
        self.assertGreater(self.store.version(A)['next_try'],time.time())

    def test_source_path_cannot_escape_registry(self):
        self.store.upsert(self.status)
        with self.store.connect() as db:
            db.execute("UPDATE versions SET archive='/etc/passwd',download_state='downloaded' WHERE id=?", (A,))
        self.assertIsNone(self.store.archive_file(A))


class ZipTests(unittest.TestCase):
    def test_unsafe_members_rejected_before_any_extraction(self):
        for filename, mode in [('../escape.py', 0), ('/escape.py', 0), ('folder\\escape.py', 0), ('link.py', 0o120777 << 16)]:
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as folder:
                raw = io.BytesIO()
                with zipfile.ZipFile(raw, 'w') as bundle:
                    bundle.writestr('safe.py', SOURCE)
                    info = zipfile.ZipInfo(filename)
                    info.external_attr = mode
                    bundle.writestr(info, SOURCE)
                with self.assertRaises(ValueError):
                    unpack_source(raw.getvalue(), Path(folder) / 'source')
                self.assertFalse((Path(folder) / 'source/safe.py').exists())

    def test_nested_sources_are_preserved_without_importing(self):
        raw = io.BytesIO()
        with zipfile.ZipFile(raw, 'w') as bundle:
            bundle.writestr('agent/main.py', SOURCE)
        with tempfile.TemporaryDirectory() as folder:
            artifact, members = unpack_source(raw.getvalue(), Path(folder))
            self.assertEqual(artifact, 'agent_code.zip')
            self.assertEqual(members, ['agent/main.py'])
            self.assertEqual((Path(folder) / 'agent/main.py').read_bytes(), SOURCE)


class ArtifactAPITests(unittest.TestCase):
    @patch('server.time.sleep')
    @patch('server.urlopen')
    def test_artifact_gate_and_429_share_global_backoff(self, request, sleep):
        from urllib.error import HTTPError
        request.side_effect = HTTPError('url', 429, 'limited', {'Retry-After': '80'}, io.BytesIO(b'{"error_code":"ARTIFACT_NOT_RELEASED"}'))
        api = server.PublicAPI()
        with self.assertRaises(RuntimeError) as caught:
            api.artifact_url(A)
        self.assertTrue(caught.exception.release_gated)
        self.assertEqual(api.get('top')['status'], 429)
        self.assertEqual(request.call_count, 1)


if __name__ == '__main__':
    unittest.main()
