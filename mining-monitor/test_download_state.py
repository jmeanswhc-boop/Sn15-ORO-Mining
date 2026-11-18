"""Offline scheduling, real progress and persistent completion-event checks."""
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch, MagicMock

from registry import Registry, Worker, utc
from race_archive import RaceArchive
from test_race_archive import A, B, RACES, RUNS, EPISODE, SOURCE, FakeAPI


class DownloadStateTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.store=Registry(self.temp.name);self.api=FakeAPI();self.archive=RaceArchive(self.store,self.api)
        self.state=self.archive.scheduling
        self.store.set_meta('suites',[{'suite_id':160,'execution_kind':'envpack'}])
        self.races=[]
        for i,ident in enumerate(RACES):
            race={'race_id':ident,'suite_id':160,'race_number':i+1,
                  'status':'RACE_COMPLETE' if i<2 else 'RACE_RUNNING',
                  'qualifying_closes_at':f'2020-01-0{1 if i==0 else 2 if i==1 else 6}T19:00:00',
                  'race_completed_at':f'2020-01-0{i+2}T03:00:00' if i<2 else None,
                  'winner_agent_version_id':B if i==2 else A}
            self.races.append(race)
            participant={'agent_version_id':A if i==0 else B,'agent_name':'Example','version_number':1}
            self.store.record_race({'race':race,'qualifiers':[participant]})
        self.archive.seed();self.store.set_meta('race_scan',{'completed_at':utc()});self.state.ensure_batch()

    def materialize(self,ident):
        with self.store.connect() as db:
            row=dict(db.execute('SELECT a.*,r.data,r.checked FROM race_archives a JOIN races r ON a.id=r.id WHERE a.id=?',(ident,)).fetchone())
        self.archive.materialize(row)

    def source(self,version):
        folder=Path(self.temp.name)/'champions'/version;folder.mkdir(parents=True)
        (folder/'agent.py').write_bytes(SOURCE)
        with self.store.connect() as db:db.execute("UPDATE versions SET archive=?,sha256=?,download_state='downloaded' WHERE id=?",
            (str((folder/'agent.py').relative_to(self.temp.name)),hashlib.sha256(SOURCE).hexdigest(),version))

    def all_old_ready(self,hidden=False):
        self.source(A)
        if hidden:
            with self.store.connect() as db:db.execute("UPDATE versions SET download_state='waiting_release',next_try=? WHERE id=?",(time.time()+86400,B))
        else:self.source(B)
        for ident in RACES:self.materialize(ident)
        for i,version in enumerate([A,B]):
            self.api.paths['agent-versions/'+version+'/runs']=[{'agent_version_id':version,'eval_run_id':RUNS[i],
                'race_id':RACES[i],'phase':'RACE','status':'SUCCESS'}]
            self.api.paths['evaluation-runs/'+RUNS[i]]={'agent_version_id':version,'eval_run_id':RUNS[i],
                'items':{'detail_released':True,'expected_item_count':0,'persisted_item_count':0,'items':[]}}
        for ident in RACES:
            self.api.paths['races/'+ident+'/seed-commitment']={'released':True}
            self.api.paths['races/'+ident+'/validator-variance']={}
        self.api.paths['suites/160/problems']={'problems':[]}
        def not_published(*args,**kwargs):
            error=RuntimeError('public artifact not found');error.not_published=True;raise error
        self.api.artifact_url=not_published
        for _ in range(4):
            with self.store.connect() as db:jobs=[dict(r) for r in db.execute("SELECT * FROM archive_assets WHERE state='pending'")]
            if not jobs:break
            for job in jobs:self.archive.download(job)
        for ident in RACES:self.materialize(ident)

    def conclude(self):
        race=dict(self.races[2],status='RACE_COMPLETE',race_completed_at='2020-01-07T03:00:00')
        self.store.record_race({'race':race,'qualifiers':[{'agent_version_id':B,'agent_name':'Example','version_number':1}]})

    def test_initial_progress_is_not_complete_with_pending_code_or_indexing(self):
        result=self.state.update()
        self.assertNotEqual(result['state'],'complete');self.assertLess(result['percent'],100)
        self.assertEqual(result['races'],2);self.assertEqual(result['events'],[])
        self.assertGreater(result['pending'],0)

    def test_batch_waits_for_full_race_history_scan(self):
        with self.store.connect() as db:
            db.execute('DELETE FROM archive_batch_races');db.execute('DELETE FROM archive_batches')
        self.store.set_meta('race_scan',{'offset':100})
        result=self.state.update();self.assertEqual(result['state'],'indexing');self.assertIsNone(result['percent'])
        self.assertEqual(result['events'],[])

    def test_completion_event_is_persistent_and_emitted_once(self):
        self.all_old_ready();first=self.state.update()
        self.assertEqual(first['state'],'complete');self.assertEqual(first['percent'],100)
        self.assertGreater(first['downloaded'],0);self.assertGreater(first['unpublished'],0)
        self.assertEqual(len(first['events']),1)
        resumed=RaceArchive(Registry(self.temp.name),self.api).scheduling
        again=resumed.update();self.assertEqual(again['events'],first['events'])
        with self.store.connect() as db:db.execute('UPDATE race_archives SET dirty=dirty+1 WHERE id=?',(RACES[0],))
        recheck=resumed.update();self.assertEqual(recheck['state'],'complete');self.assertEqual(len(recheck['events']),1)

    def test_hidden_code_is_explicitly_reported_as_waiting_not_downloaded(self):
        self.all_old_ready(hidden=True);result=self.state.update()
        self.assertEqual(result['state'],'complete');self.assertEqual(result['waiting_release'],1)
        self.assertIn('await public release',result['events'][0]['message'])

    def test_partial_or_unavailable_diagnostics_prevent_completion(self):
        self.all_old_ready()
        ident=self.archive.queue('feedback',version=A,run=RUNS[0],race=RACES[0],episode=EPISODE)
        for reason in ['PARTIAL_RESULT','DIAGNOSTICS_UNAVAILABLE']:
            with self.store.connect() as db:db.execute("UPDATE archive_assets SET state='waiting_release',error=? WHERE id=?",(reason,ident))
            result=self.state.update();self.assertNotEqual(result['state'],'complete');self.assertGreater(result['pending'],0)
            self.assertEqual(result['events'],[])

    def test_error_is_not_counted_as_success(self):
        self.all_old_ready();ident=self.archive.queue('feedback',version=A,run=RUNS[0],race=RACES[0],episode=EPISODE)
        with self.store.connect() as db:db.execute("UPDATE archive_assets SET state='error' WHERE id=?",(ident,))
        result=self.state.update();self.assertEqual(result['errors'],1);self.assertLess(result['percent'],100)
        self.assertEqual(result['events'],[])

    def test_new_race_promotes_winner_and_five_day_release_cohort(self):
        self.conclude();source=self.state.source()
        self.assertEqual(source['id'],B);self.assertEqual(source['priority_race'],RACES[2])
        self.source(B);source=self.state.source()
        self.assertEqual(source['id'],A);self.assertEqual(source['priority_race'],RACES[0])
        with self.store.connect() as db:
            scopes=dict(db.execute('SELECT race,scope FROM archive_priorities'))
            members=[r[0] for r in db.execute('SELECT race FROM archive_batch_races')]
        self.assertEqual(scopes[RACES[2]],'all');self.assertEqual(scopes[RACES[0]],'code')
        self.assertNotIn(RACES[2],members)

    def test_new_priority_runs_before_old_work_and_round_robin_resumes(self):
        self.conclude()
        with self.store.connect() as db:
            db.execute('UPDATE archive_assets SET next_try=?',(time.time()+3600,))
            db.execute("UPDATE versions SET download_state='waiting_release',next_try=?",(time.time()+3600,))
            db.execute('UPDATE race_archives SET dirty=0,synced=(SELECT checked FROM races WHERE races.id=race_archives.id),materialized_at=?',(time.time(),))
        older=self.archive.queue('feedback',version=A,run=RUNS[0],race=RACES[1],episode=EPISODE)
        newer=self.archive.queue('feedback',version=B,run=RUNS[2],race=RACES[2],episode=EPISODE)
        self.archive.job_turn=2
        with patch.object(self.archive,'download') as download:
            self.archive.tick();self.assertEqual(download.call_args[0][0]['id'],newer)
        self.assertEqual(self.archive.job_turn,2)
        with self.store.connect() as db:db.execute('UPDATE archive_assets SET next_try=? WHERE id=?',(time.time()+3600,newer))
        with patch.object(self.archive,'download') as download:
            self.archive.tick();self.assertEqual(download.call_args[0][0]['id'],older)
        self.assertEqual(self.archive.job_turn,3)
        resumed=RaceArchive(Registry(self.temp.name),self.api)
        self.assertEqual(resumed.job_turn,3)

    def test_latest_of_two_completed_races_takes_precedence(self):
        self.conclude()
        with self.store.connect() as db:
            later=dict(self.races[1],status='RACE_COMPLETE',race_completed_at='2020-01-08T03:00:00')
            self.state.promote(db,later)
        self.assertEqual(self.state.source()['priority_race'],RACES[1])

    def test_worker_refreshes_new_roster_before_historical_cursor_advances(self):
        self.conclude();self.store.set_meta('turn',37)
        self.api.paths['races/'+RACES[2]]={'race':dict(self.races[2],status='RACE_COMPLETE'),
            'qualifiers':[{'agent_version_id':B,'agent_name':'Example','version_number':1}]}
        worker=Worker(self.store,self.api);worker.step()
        self.assertEqual(self.api.calls,['races/'+RACES[2]])
        self.assertEqual(self.store.meta('turn'),37)
        with self.store.connect() as db:self.assertGreater(db.execute('SELECT checked FROM races WHERE id=?',(RACES[2],)).fetchone()[0],0)

    def test_priority_run_list_invalidates_pre_completion_cache(self):
        for ident in RACES:self.materialize(ident)
        self.conclude();job=self.state.asset()
        self.assertEqual(job['kind'],'run_list')
        self.api.invalidate=MagicMock()
        self.api.paths['agent-versions/'+job['version']+'/runs']=[]
        self.archive.download(job)
        self.api.invalidate.assert_called_once_with('agent-versions/'+job['version']+'/runs')

    def test_multiple_dirty_events_are_materialized(self):
        for ident in RACES:self.materialize(ident)
        self.source(A);self.archive.mark_dirty(A);self.archive.mark_dirty(A)
        with self.store.connect() as db:db.execute('UPDATE race_archives SET materialized_at=0 WHERE id=?',(RACES[0],))
        self.archive.tick()
        m=self.archive.race_manifest(RACES[0]);self.assertEqual(m['code_counts']['downloaded'],1)

if __name__=='__main__':unittest.main()
