"""Race archival contracts, using local fixtures and mocked public reads only."""
import hashlib
import gzip
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from registry import Registry, unpack_source
from race_archive import RaceArchive

A='12345678-1234-1234-1234-123456789abc'
B='22345678-1234-1234-1234-123456789abc'
RACES=['33333333-3333-3333-3333-33333333333'+str(i) for i in range(3)]
RUNS=['44444444-4444-4444-4444-44444444444'+str(i) for i in range(3)]
EPISODE='55555555-5555-5555-5555-555555555555'
SOURCE=b'def agent_main(problem_data):\n    return {}\n'


class FakeAPI:
    def __init__(self):self.paths={};self.calls=[]
    def get(self,path):
        self.calls.append(path)
        return {'data':self.paths[path],'stale':False,'error':None}
    def artifact_url(self,*args,**kwargs):raise AssertionError('Unexpected artifact fetch')


class RaceArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.store=Registry(self.temp.name);self.api=FakeAPI();self.archive=RaceArchive(self.store,self.api)
        self.store.set_meta('suites',[{'suite_id':160,'execution_kind':'envpack'}])
        for number,ident in enumerate(RACES,1):
            self.store.record_race({'race':{'race_id':ident,'suite_id':160,'race_number':number,
                'status':'RACE_COMPLETE','qualifying_closes_at':f'2026-01-0{number}T00:00:00'},
                'qualifiers':[{'agent_version_id':A,'agent_name':'Example','version_number':1,
                    'miner_hotkey':'5'+'A'*47,'race_score':number/10,'qualifying_score':.8}]})
        self.archive.seed()

    def materialize(self,ident):
        with self.store.connect() as db:
            row=db.execute('SELECT a.*,r.data,r.checked FROM race_archives a JOIN races r ON a.id=r.id WHERE a.id=?',(ident,)).fetchone()
        self.archive.materialize(dict(row))
        return self.archive.race_manifest(ident)

    def source(self):
        folder=Path(self.temp.name)/'champions/example';(folder/'source').mkdir(parents=True)
        (folder/'agent.py').write_bytes(SOURCE);(folder/'source/agent.py').write_bytes(SOURCE)
        (folder/'metadata.json').write_text('{}')
        digest=hashlib.sha256(SOURCE).hexdigest()
        with self.store.connect() as db:db.execute("UPDATE versions SET archive=?,sha256=?,download_state='downloaded' WHERE id=?",('champions/example/agent.py',digest,A))
        return folder/'agent.py'

    def job(self,kind,**kwargs):
        ident=self.archive.queue(kind,version=A,**kwargs)
        with self.store.connect() as db:return dict(db.execute('SELECT * FROM archive_assets WHERE id=?',(ident,)).fetchone())

    def test_same_agent_is_independently_available_in_three_races(self):
        canonical=self.source();manifests=[self.materialize(r) for r in RACES]
        paths=[Path(self.temp.name)/m['directory']/m['participants'][0]['code_file'] for m in manifests]
        self.assertEqual(len(set(paths)),3)
        self.assertTrue(all(p.read_bytes()==SOURCE for p in paths))
        self.assertEqual([m['participants'][0]['result']['race_score'] for m in manifests],[.1,.2,.3])
        paths[0].write_bytes(b'local edit')
        self.assertEqual(paths[1].read_bytes(),SOURCE);self.assertEqual(canonical.read_bytes(),SOURCE)
        self.assertEqual(len({p.stat().st_ino for p in paths+[canonical]}),4)
        with self.store.connect() as db:self.assertEqual(db.execute("SELECT COUNT(*) FROM archive_assets WHERE kind='run_list'").fetchone()[0],1)

    def test_runs_are_filtered_locally_even_when_endpoint_returns_all_races(self):
        runs=[{'agent_version_id':A,'eval_run_id':run,'race_id':race,'phase':'RACE','status':'SUCCESS'} for run,race in zip(RUNS,RACES)]
        runs.append({'agent_version_id':A,'eval_run_id':B,'phase':'QUALIFYING','status':'SUCCESS'})
        self.api.paths['agent-versions/'+A+'/runs']=runs
        job=self.job('run_list');self.archive.download(job)
        for race,run in zip(RACES,RUNS):
            m=self.materialize(race)
            data=json.loads((Path(self.temp.name)/m['directory']/'agents'/A/'runs.json').read_text())
            self.assertEqual([r['eval_run_id'] for r in data],[run])
            entry=next(a for a in m['assets'] if a['kind']=='run_list')
            path=Path(self.temp.name)/m['directory']/entry['file']
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),entry['sha256'])
            self.assertIn('source_sha256',entry)
            self.assertTrue(all(a['run']==run for a in m['assets'] if a['run']))

    def test_embargoed_run_is_retryable_and_not_complete(self):
        self.source();job=self.job('run_detail',run=RUNS[0],race=RACES[0])
        self.api.paths['evaluation-runs/'+RUNS[0]]={'items':{'detail_released':False}}
        self.archive.download(job)
        with self.store.connect() as db:row=db.execute('SELECT * FROM archive_assets WHERE id=?',(job['id'],)).fetchone()
        self.assertEqual(row['state'],'waiting_release');self.assertGreater(row['next_try'],time.time())
        m=self.materialize(RACES[0]);self.assertEqual(m['state'],'collecting')

    def test_partial_result_saves_snapshot_but_keeps_retrying(self):
        job=self.job('run_detail',run=RUNS[0],race=RACES[0])
        self.api.paths['evaluation-runs/'+RUNS[0]]={'agent_version_id':A,'eval_run_id':RUNS[0],
            'items':{'detail_released':True,'expected_item_count':90,'persisted_item_count':1,
                     'items':[{'episode_result_id':EPISODE}]}}
        self.archive.download(job)
        with self.store.connect() as db:
            row=db.execute('SELECT * FROM archive_assets WHERE id=?',(job['id'],)).fetchone()
            count=db.execute("SELECT COUNT(*) FROM archive_assets WHERE kind='feedback'").fetchone()[0]
        self.assertEqual(row['state'],'waiting_release');self.assertTrue((Path(self.temp.name)/row['path']).exists())
        self.assertEqual(count,1)

    def test_unavailable_diagnostics_are_not_counted_as_downloaded(self):
        job=self.job('feedback',run=RUNS[0],race=RACES[0],episode=EPISODE)
        self.api.paths['episode-results/'+EPISODE+'/feedback']={'diagnostics_state':'diagnostics_unavailable'}
        self.archive.download(job)
        with self.store.connect() as db:row=db.execute('SELECT * FROM archive_assets WHERE id=?',(job['id'],)).fetchone()
        self.assertEqual(row['state'],'waiting_release');self.assertIsNone(row['path'])

    def test_not_published_is_distinct_from_release_gate_and_error(self):
        job=self.job('log_bundle',run=RUNS[0],race=RACES[0]);error=RuntimeError('not found');error.not_published=True
        with patch.object(self.api,'artifact_url',side_effect=error):self.archive.download(job)
        with self.store.connect() as db:row=db.execute('SELECT * FROM archive_assets WHERE id=?',(job['id'],)).fetchone()
        self.assertEqual(row['state'],'unpublished');self.assertGreater(row['next_try'],time.time())

    def test_wrong_run_identity_is_not_archived(self):
        job=self.job('run_detail',run=RUNS[0],race=RACES[0])
        self.api.paths['evaluation-runs/'+RUNS[0]]={'agent_version_id':B,'eval_run_id':RUNS[0],'items':{}}
        self.archive.download(job)
        with self.store.connect() as db:row=db.execute('SELECT * FROM archive_assets WHERE id=?',(job['id'],)).fetchone()
        self.assertEqual(row['state'],'error');self.assertIsNone(row['path'])

    def test_restart_preserves_jobs_and_files_without_duplicates(self):
        self.source();self.materialize(RACES[0]);before=self.archive.snapshot()
        resumed=RaceArchive(Registry(self.temp.name),self.api);resumed.seed();after=resumed.snapshot()
        self.assertEqual(before['assets'],after['assets']);self.assertEqual(len(after['races']),3)
        self.assertTrue(resumed.file(RACES[0],'agents/'+A+'/agent.py').exists())

    def test_race_file_route_rejects_escape_and_symlink(self):
        self.source();m=self.materialize(RACES[0]);outside=Path(self.temp.name)/'outside.txt';outside.write_text('private')
        path=Path(self.temp.name)/m['directory']/'escape';path.symlink_to(outside)
        self.assertIsNone(self.archive.file(RACES[0],'../../outside.txt'))
        self.assertIsNone(self.archive.file(RACES[0],'escape'))
        self.assertIsNone(self.archive.file(B,'race.json'))

    def test_roster_replacement_does_not_leave_previous_candidates(self):
        self.store.record_race({'race':{'race_id':RACES[0],'suite_id':160,'race_number':1,'status':'RACE_RUNNING'},
            'qualifiers':[{'agent_version_id':B,'agent_name':'Replacement','version_number':1}]})
        m=self.materialize(RACES[0]);self.assertEqual([r['agent_version_id'] for r in m['participants']],[B])
        self.assertEqual(self.materialize(RACES[1])['participants'][0]['agent_version_id'],A)

    def test_download_event_during_copy_is_not_lost(self):
        self.source()
        with self.store.connect() as db:row=dict(db.execute('SELECT a.*,r.data,r.checked FROM race_archives a JOIN races r ON a.id=r.id WHERE a.id=?',(RACES[0],)).fetchone())
        self.archive.mark_dirty(A);self.archive.materialize(row)
        with self.store.connect() as db:dirty=db.execute('SELECT dirty FROM race_archives WHERE id=?',(RACES[0],)).fetchone()[0]
        self.assertGreater(dirty,0)

    def test_invalid_historical_python_is_archived_without_execution(self):
        with tempfile.TemporaryDirectory() as folder:
            raw=b'def invalid(\n'
            artifact,members=unpack_source(raw,Path(folder))
            self.assertEqual(artifact,'agent.py');self.assertEqual((Path(folder)/'agent.py').read_bytes(),raw)

    def test_gzipped_artifact_stream_preserves_decoded_bytes(self):
        class Response(io.BytesIO):headers={'Content-Encoding':'gzip'}
        destination=Path(self.temp.name)/'download.incoming'
        with patch('race_archive.urlopen',return_value=Response(gzip.compress(SOURCE))):
            self.archive.fetch_artifact('https://storage.example/released',destination)
        self.assertEqual(destination.read_bytes(),SOURCE)

    def test_interrupted_artifact_is_not_marked_downloaded(self):
        job=self.job('log_bundle',run=RUNS[0],race=RACES[0])
        with patch.object(self.api,'artifact_url',return_value={'download_url':'https://storage.example/released'}):
            with patch.object(self.archive,'fetch_artifact',side_effect=TimeoutError('interrupted')):
                self.archive.download(job)
        with self.store.connect() as db:row=db.execute('SELECT * FROM archive_assets WHERE id=?',(job['id'],)).fetchone()
        self.assertEqual(row['state'],'error');self.assertIsNone(row['path'])

    def test_episode_file_has_one_physical_copy_in_its_own_race(self):
        self.archive.index_runs(A,[{'agent_version_id':A,'eval_run_id':RUNS[0],
            'race_id':RACES[0],'phase':'RACE','status':'SUCCESS'}])
        job=self.job('feedback',run=RUNS[0],race=RACES[0],episode=EPISODE)
        self.api.paths['episode-results/'+EPISODE+'/feedback']={'diagnostics_state':'available','query':None}
        self.archive.download(job);m=self.materialize(RACES[0])
        entry=next(a for a in m['assets'] if a['kind']=='feedback')
        path=Path(self.temp.name)/m['directory']/entry['file']
        self.assertTrue(path.is_file())
        self.assertEqual(len(list(Path(self.temp.name).rglob('feedback-'+EPISODE+'.json'))),1)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),entry['sha256'])

    def test_truncated_public_transfer_is_rejected(self):
        class Response(io.BytesIO):
            headers={}
            length=7
        destination=Path(self.temp.name)/'partial.incoming'
        with patch('race_archive.urlopen',return_value=Response(b'partial')):
            with self.assertRaises(EOFError):
                self.archive.fetch_artifact('https://storage.example/released',destination)


if __name__=='__main__':unittest.main()
