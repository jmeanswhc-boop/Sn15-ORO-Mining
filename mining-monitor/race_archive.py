"""Resumable race-specific copies of public ORO artifacts and result records."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import threading
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from registry import VERSION, atomic_json, epoch, utc

def file_hash(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


class RaceArchive:
    def __init__(self, store, api):
        self.store, self.api = store, api
        self.root = store.root / 'race-archives'
        self.stop = threading.Event()
        self.thread = None
        self.job_turn=store.meta('archive_job_turn',0)
        self.last_progress=0
        with store.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS race_archives (
                    id TEXT PRIMARY KEY, directory TEXT, synced REAL DEFAULT 0,
                    dirty INTEGER DEFAULT 1, manifest TEXT DEFAULT '{}',
                    materialized_at REAL DEFAULT 0, summary TEXT DEFAULT '{}');
                CREATE TABLE IF NOT EXISTS archive_runs (
                    id TEXT PRIMARY KEY, version TEXT, race TEXT, data TEXT);
                CREATE INDEX IF NOT EXISTS archive_runs_race ON archive_runs(race,version);
                CREATE TABLE IF NOT EXISTS archive_assets (
                    id TEXT PRIMARY KEY, kind TEXT, version TEXT, run TEXT, race TEXT,
                    episode TEXT, suite INTEGER, state TEXT DEFAULT 'pending',
                    path TEXT, sha256 TEXT, bytes INTEGER, next_try REAL DEFAULT 0,
                    attempts INTEGER DEFAULT 0, checked REAL DEFAULT 0, error TEXT);
                CREATE INDEX IF NOT EXISTS archive_assets_ready ON archive_assets(state,next_try);
                CREATE INDEX IF NOT EXISTS archive_assets_version ON archive_assets(version,run);
                CREATE INDEX IF NOT EXISTS archive_assets_kind ON archive_assets(kind,next_try,state);
                CREATE INDEX IF NOT EXISTS archive_assets_race ON archive_assets(race,state,next_try);
            ''')
            columns={r[1] for r in db.execute('PRAGMA table_info(race_archives)')}
            for name,declaration in [('materialized_at','REAL DEFAULT 0'),('summary',"TEXT DEFAULT '{}'")]:
                if name not in columns:db.execute(f'ALTER TABLE race_archives ADD COLUMN {name} {declaration}')
            db.execute("UPDATE versions SET download_state='pending',next_try=0 WHERE error='SyntaxError'")
        from download_state import DownloadState
        self.scheduling=DownloadState(store)

    def start(self):
        self.thread = threading.Thread(target=self.run, name='orobench-race-files', daemon=True)
        self.thread.start()

    def run(self):
        while not self.stop.is_set():
            try:
                worked = self.tick()
                if time.monotonic()-self.last_progress>=5:
                    self.last_progress=time.monotonic()
                    try:self.scheduling.update()
                    except Exception as error:
                        self.store.set_meta('download_progress',{'state':'checking','percent':None,'error':type(error).__name__})
                self.store.set_meta('race_archive_health', {'state':'running', 'updated_at':utc(), 'error':None})
            except Exception as error:
                worked = False
                self.store.set_meta('race_archive_health', {'state':'retrying', 'updated_at':utc(), 'error':type(error).__name__})
            self.stop.wait(.2 if worked else 3)

    def queue(self, kind, version='', run='', race='', episode='', suite=0, connection=None):
        from contextlib import nullcontext
        ident = hashlib.sha256(json.dumps([kind,version,run,race,episode,suite]).encode()).hexdigest()
        with (nullcontext(connection) if connection is not None else self.store.connect()) as db:
            db.execute('''INSERT OR IGNORE INTO archive_assets
                (id,kind,version,run,race,episode,suite) VALUES (?,?,?,?,?,?,?)''',
                       (ident,kind,version,run,race,episode,suite))
        return ident

    def mark_dirty(self, version):
        with self.store.connect() as db:
            db.execute('UPDATE race_archives SET dirty=dirty+1 WHERE id IN (SELECT race FROM results WHERE version=?)', (version,))

    def seed(self):
        with self.store.connect() as db:
            races = db.execute('SELECT * FROM races').fetchall()
            known = {r[0] for r in db.execute('SELECT id FROM race_archives')}
            for row in races:
                if row['id'] in known:
                    continue
                race = json.loads(row['data'])
                directory = f"race-archives/suite-{row['suite']}/race-{race['race_number']:03d}-{row['id']}"
                db.execute('INSERT INTO race_archives(id,directory) VALUES (?,?)', (row['id'],directory))
                self.queue('race_seed', race=row['id'], connection=db)
                self.queue('race_variance', race=row['id'], connection=db)
                self.queue('suite_qualifiers', suite=row['suite'], connection=db)

    def get(self, path, fresh=False):
        if fresh and hasattr(self.api,'invalidate'):self.api.invalidate(path)
        result = self.api.get(path)
        if result.get('error') or result.get('stale') or result.get('data') is None:
            error = RuntimeError('Public data unavailable')
            error.retry_after = result.get('retry_after', 60)
            error.release_gated = result.get('status')==403
            error.not_published = result.get('status')==404
            raise error
        return result['data']

    def tick(self):
        self.seed()
        self.scheduling.ensure_batch()
        self.scheduling.bootstrap_priority()
        urgent=self.scheduling.materialization()
        if urgent:
            self.materialize(urgent);return True
        job=self.scheduling.asset()
        if job:
            self.download(job);return True
        if self.scheduling.source():return False
        # Copy completed downloads before taking the next network job.
        with self.store.connect() as db:
            race = db.execute('''SELECT a.*,r.data,r.checked FROM race_archives a JOIN races r ON a.id=r.id
                WHERE (a.dirty>0 OR a.synced<r.checked) AND a.materialized_at<?
                ORDER BY a.synced,a.id LIMIT 1''',(time.time()-15,)).fetchone()
        if race:
            self.materialize(dict(race))
            return True
        # Rotate resource kinds so an expanding episode queue or missing log
        # bundles cannot prevent new race metadata and released files from advancing.
        kinds=['run_list','run_detail','feedback','log_bundle','feedback','race_seed',
               'run_detail','reasoning','feedback','suite_qualifiers','race_variance','problem_logs']
        job=None
        with self.store.connect() as db:
            for delta in range(len(kinds)):
                slot=(self.job_turn+delta)%len(kinds)
                job=db.execute('''SELECT * FROM archive_assets WHERE kind=?
                    AND (state!='downloaded' OR kind='run_list') AND next_try<=?
                    ORDER BY CASE state WHEN 'pending' THEN 0 WHEN 'waiting_release' THEN 1
                      WHEN 'error' THEN 2 WHEN 'unpublished' THEN 3 ELSE 4 END,checked,id LIMIT 1''',
                    (kinds[slot],time.time())).fetchone()
                if job:
                    self.job_turn=slot+1;break
        if job:
            self.store.set_meta('archive_job_turn',self.job_turn)
            self.download(dict(job))
            return True
        return False

    def materialize(self, row):
        race = json.loads(row['data'])
        folder = self.store.root / row['directory']
        folder.mkdir(parents=True, exist_ok=True)
        with self.store.connect() as db:
            participants = db.execute('''SELECT v.*,r.data AS participation FROM results r
                JOIN versions v ON v.id=r.version WHERE r.race=? ORDER BY v.id''', (row['id'],)).fetchall()
            assets = db.execute('''SELECT a.* FROM archive_assets a WHERE
                a.race=? OR (a.kind='suite_qualifiers' AND a.suite=?) OR a.version IN
                (SELECT version FROM results WHERE race=?)''', (row['id'],row['suite'] if 'suite' in row else race['suite_id'],row['id'])).fetchall()
            runs = [dict(r) for r in db.execute('SELECT * FROM archive_runs WHERE race=?', (row['id'],))]
        run_ids = {r['id'] for r in runs}
        assets = [dict(a) for a in assets if not a['run'] or a['run'] in run_ids]
        by_version = {}
        for asset in assets:
            by_version.setdefault(asset['version'], []).append(asset)
        records = []
        code_counts = {'downloaded':0,'pending':0,'waiting_release':0,'error':0}
        file_counts = {'downloaded':0,'pending':0,'waiting_release':0,'unpublished':0,'error':0}
        manifest_assets = []
        for participant in participants:
            value = json.loads(participant['data'])
            value.pop('runs', None)
            ident = participant['id']
            agent_folder = folder / 'agents' / ident
            agent_folder.mkdir(parents=True, exist_ok=True)
            participation = json.loads(participant['participation'])
            atomic_json(agent_folder / 'participant.json', {'race_id':row['id'], 'race_number':race['race_number'],
                        'suite_id':race['suite_id'], 'agent':value, 'result':participation})
            artifact = self.store.archive_file(ident)
            code_state = participant['download_state']
            copied = None
            if artifact:
                copied = agent_folder / artifact.name
                self.copy_verified(artifact, copied, participant['sha256'])
                metadata = artifact.parent / 'metadata.json'
                if metadata.is_file():
                    self.copy_verified(metadata, agent_folder / 'code_metadata.json')
                # Each race has independent copies; editing one cannot alter another race.
                source = artifact.parent / 'source'
                if source.is_dir() and not (agent_folder / 'source').exists():
                    temporary = agent_folder / '.source-incoming'
                    if temporary.exists():
                        shutil.rmtree(temporary)
                    shutil.copytree(source, temporary, symlinks=False)
                    temporary.replace(agent_folder / 'source')
                code_state = 'downloaded'
            elif code_state == 'downloaded':
                code_state = 'error'
            code_counts[code_state] = code_counts.get(code_state,0)+1
            records.append({'agent_version_id':ident, 'agent_name':value.get('agent_name'),
                            'version_number':value.get('version_number'), 'miner_hotkey':value.get('miner_hotkey'),
                            'result':participation, 'code_state':code_state, 'code_sha256':participant['sha256'],
                            'code_file':str(copied.relative_to(folder)) if copied else None,
                            'code_available_at':value.get('code_available_at'), 'code_error':participant['error']})
            self.queue('run_list', version=ident)
            agent_runs = [json.loads(r['data']) for r in runs if r['version']==ident]
            atomic_json(agent_folder / 'runs.json', agent_runs)
        for asset in assets:
            file_counts[asset['state']] = file_counts.get(asset['state'],0)+1
            entry = {k:asset[k] for k in ('id','kind','version','run','episode','state','sha256','bytes','error','next_try')}
            if asset['state']=='downloaded' and asset['path']:
                source = (self.store.root / asset['path']).resolve()
                if not source.is_relative_to(self.root) or not source.is_file():
                    entry['state']='error'
                    file_counts['downloaded']-=1;file_counts['error']+=1
                else:
                    if asset['version']:
                        destination = folder / 'agents' / asset['version']
                        if asset['run']:
                            destination = destination / 'runs' / asset['run']
                    else:
                        destination = folder / 'race-data'
                    # run-list objects contain multiple races; runs.json above is the filtered view.
                    if asset['kind']=='run_list':
                        destination=destination/'runs.json'
                        entry['source_sha256']=entry['sha256']
                        entry['sha256']=file_hash(destination)
                        entry['bytes']=destination.stat().st_size
                        entry['file']=str(destination.relative_to(folder))
                    else:
                        destination = destination / source.name
                        self.copy_verified(source, destination, asset['sha256'])
                        entry['file'] = str(destination.relative_to(folder))
            manifest_assets.append(entry)
        pending_codes = sum(v for k,v in code_counts.items() if k!='downloaded')
        unfinished = sum(file_counts.get(k,0) for k in ('pending','waiting_release','error'))
        enumerated = sum(1 for a in assets if a['kind']=='run_list' and a['state']=='downloaded')
        covered_versions={r['version'] for r in runs}
        manifest = {'schema_version':1,'updated_at':utc(), 'race':race, 'directory':row['directory'],
                    'roster_fetched_at':row['checked'], 'participants':records,'assets':manifest_assets,
                    'code_counts':code_counts,'file_counts':file_counts,'participant_count':len(records),
                    'run_lists_indexed':enumerated,'run_count':len(runs),
                    'state':'complete' if race.get('status')=='RACE_COMPLETE' and participants and not pending_codes
                        and not unfinished and enumerated==len(participants) and len(covered_versions)==len(participants)
                        and all(json.loads(r['data']).get('status') in {'SUCCESS','FAILED','TIMED_OUT','CANCELLED','STALE'} for r in runs) else 'collecting',
                    'scope':'Public code, race metadata, public run details, published logs and episode feedback. Private race worlds are excluded.'}
        atomic_json(folder / 'race.json', race)
        atomic_json(folder / 'roster.json', {'race':race,'qualifiers':[json.loads(p['participation']) for p in participants]})
        atomic_json(folder / 'manifest.json', manifest)
        temporary = folder / 'participants.csv.tmp'
        with temporary.open('w',newline='') as output:
            writer=csv.writer(output);writer.writerow(['agent_version_id','agent_name','version_number','code_state','race_score','overall_score','rank'])
            for value in records:
                result=value['result'];fields=[value['agent_version_id'],value['agent_name'],value['version_number'],
                    value['code_state'],result.get('race_score'),result.get('weighted_score'),result.get('race_rank')]
                writer.writerow(["'"+v if isinstance(v,str) and v.startswith(('=','+','-','@','\t','\r')) else v for v in fields])
        temporary.replace(folder/'participants.csv')
        with self.store.connect() as db:
            summary={k:v for k,v in manifest.items() if k not in {'participants','assets'}}
            # Subtract only changes seen at the start; concurrent downloads remain dirty.
            db.execute('UPDATE race_archives SET manifest=?,summary=?,synced=?,dirty=MAX(0,dirty-?),materialized_at=? WHERE id=?',
                       (json.dumps(manifest),json.dumps(summary),row['checked'],row.get('dirty',1),time.time(),row['id']))
        self.export_index()

    @staticmethod
    def copy_verified(source, destination, digest=None):
        source_stat=source.stat()
        if destination.is_file():
            target_stat=destination.stat()
            if target_stat.st_size==source_stat.st_size and target_stat.st_mtime_ns==source_stat.st_mtime_ns:
                return
        digest = digest or file_hash(source)
        if destination.is_file() and file_hash(destination)==digest:
            shutil.copystat(source,destination)
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary=destination.with_name(destination.name+'.incoming')
        shutil.copyfile(source,temporary)
        if file_hash(temporary)!=digest:
            raise ValueError('Race copy failed SHA-256 verification')
        shutil.copystat(source,temporary)
        temporary.replace(destination)

    def download(self, job):
        kind=job['kind'];version=job['version'];run=job['run'];race=job['race'];episode=job['episode']
        folder=self.root/'_objects'/job['id']
        if run and race:
            with self.store.connect() as db:
                destination=db.execute('SELECT directory FROM race_archives WHERE id=?',(race,)).fetchone()
            if destination:
                # Run/episode objects belong to exactly one race. Store them once,
                # directly in that race; only cross-race resources need a cache.
                folder=self.store.root/destination[0]/'agents'/version/'runs'/run
        try:
            if shutil.disk_usage(self.store.root).free < 2*1024**3:
                raise OSError('Insufficient archive disk space')
            if kind=='run_list':
                data=self.get('agent-versions/'+version+'/runs',fresh=bool(job.get('priority_race')))
                self.index_runs(version,data)
            elif kind=='run_detail':
                data=self.get('evaluation-runs/'+run)
                items=data.get('items') or {}
                if not items.get('detail_released', True):
                    self.defer(job,'waiting_release',60,'RACE_IN_FLIGHT');return
                self.index_details(job,data)
            elif kind=='feedback':
                data=self.get('episode-results/'+episode+'/feedback')
                if data.get('diagnostics_state')!='available':
                    self.defer(job,'waiting_release',300,'DIAGNOSTICS_UNAVAILABLE');return
            elif kind=='suite_qualifiers':
                data=self.get(f"suites/{job['suite']}/problems")
            elif kind=='race_seed':
                data=self.get('races/'+race+'/seed-commitment')
            elif kind=='race_variance':
                data=self.get('races/'+race+'/validator-variance')
            else:
                artifact_type={'log_bundle':'EVAL_LOGS_BUNDLE','reasoning':'REASONING_DETAILS','problem_logs':'EVAL_PROBLEM_LOGS'}[kind]
                metadata=self.api.artifact_url(version,artifact_type=artifact_type,eval_run_id=run,
                                               problem_id=episode or None)
                folder.mkdir(parents=True,exist_ok=True)
                stem=kind+('-'+episode if episode else '')
                temporary=folder/(stem+'.incoming')
                self.fetch_artifact(metadata['download_url'],temporary)
                with temporary.open('rb') as stream:prefix=stream.read(4)
                path=folder/(stem+('.zip' if prefix.startswith(b'PK') else '.bin'))
                temporary.replace(path)
                self.finish(job,path);return
            folder.mkdir(parents=True,exist_ok=True)
            path=folder/(kind+('-'+episode if episode else '')+'.json')
            atomic_json(path,data)
            self.finish(job,path,path.read_bytes())
            if kind=='run_detail' and self.store.meta('partial_run:'+run,False):
                self.defer(job,'waiting_release',300,'PARTIAL_RESULT')
        except Exception as error:
            gated=getattr(error,'release_gated',False)
            not_found=getattr(error,'not_published',False)
            self.defer(job,'waiting_release' if gated else 'unpublished' if not_found else 'error',
                       max(getattr(error,'retry_after',0),3600 if not_found else min(3600,30*2**min(job['attempts'],6))),
                       getattr(error,'public_code',None) or type(error).__name__)

    @staticmethod
    def fetch_artifact(url,destination):
        parsed=urlsplit(url)
        if parsed.scheme!='https' or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError('Unsupported public artifact URL')
        with urlopen(Request(url,headers={'Accept-Encoding':'identity'}),timeout=30) as response:
            stream=gzip.GzipFile(fileobj=response) if response.headers.get('Content-Encoding')=='gzip' else response
            with destination.open('wb') as output:
                while True:
                    chunk=stream.read(1024*1024)
                    if not chunk:break
                    if shutil.disk_usage(destination.parent).free < 2*1024**3:
                        raise OSError('Insufficient archive disk space')
                    output.write(chunk)
                if getattr(response,'length',None) not in (None,0):
                    raise EOFError('Public artifact transfer ended before Content-Length')

    def index_runs(self, version, runs):
        with self.store.connect() as db:
            known={r[0] for r in db.execute('SELECT id FROM races')}
            for run in runs:
                if run.get('phase')!='RACE' or run.get('race_id') not in known:
                    continue
                if run.get('agent_version_id')!=version or not VERSION.fullmatch(run.get('eval_run_id','')):
                    continue
                ident=run['eval_run_id']
                db.execute('INSERT OR REPLACE INTO archive_runs VALUES (?,?,?,?)',
                           (ident,version,run['race_id'],json.dumps(run)))
                if run.get('status') in {'SUCCESS','FAILED','TIMED_OUT','CANCELLED','STALE'}:
                    self.queue('run_detail',version=version,run=ident,race=run['race_id'],connection=db)
                    self.queue('log_bundle',version=version,run=ident,race=run['race_id'],connection=db)
                    self.queue('reasoning',version=version,run=ident,race=run['race_id'],connection=db)

    def index_details(self, job, data):
        if data.get('eval_run_id')!=job['run'] or data.get('agent_version_id')!=job['version']:
            raise ValueError('Public run identity mismatch')
        items=data.get('items') or {}
        values=items.get('items',[]) if isinstance(items,dict) else []
        if isinstance(items,dict) and items.get('expected_item_count',0)>items.get('persisted_item_count',0):
            # Partial records are saved but remain retryable until ORO has persisted the roster.
            self.store.set_meta('partial_run:'+job['run'],True)
        else:
            self.store.set_meta('partial_run:'+job['run'],False)
        with self.store.connect() as db:
            for item in values:
                episode=item.get('episode_result_id')
                if episode and VERSION.fullmatch(episode):
                    self.queue('feedback',version=job['version'],run=job['run'],race=job['race'],episode=episode,connection=db)
                problem=item.get('problem_id')
                if problem and VERSION.fullmatch(problem):
                    self.queue('problem_logs',version=job['version'],run=job['run'],race=job['race'],episode=problem,connection=db)

    def finish(self, job, path, raw=None):
        digest=hashlib.sha256(raw).hexdigest() if raw is not None else file_hash(path)
        size=len(raw) if raw is not None else path.stat().st_size
        with self.store.connect() as db:
            db.execute("UPDATE archive_assets SET state='downloaded',path=?,sha256=?,bytes=?,checked=?,next_try=?,error=NULL WHERE id=?",
                       (str(path.relative_to(self.store.root)),digest,size,time.time(),
                        time.time()+86400 if job['kind']=='run_list' else 0,job['id']))
        self.mark_dirty(job['version'])
        if job['race'] or job['kind']=='suite_qualifiers':
            with self.store.connect() as db:
                db.execute('UPDATE race_archives SET dirty=dirty+1 WHERE id=? OR id IN (SELECT id FROM races WHERE suite=?)',
                           (job['race'],job['suite']))

    def defer(self, job, state, delay, error):
        with self.store.connect() as db:
            db.execute('UPDATE archive_assets SET state=?,next_try=?,attempts=attempts+1,checked=?,error=? WHERE id=?',
                       (state,time.time()+delay,time.time(),error,job['id']))
        self.mark_dirty(job['version'])
        if job['race']:
            with self.store.connect() as db:db.execute('UPDATE race_archives SET dirty=dirty+1 WHERE id=?',(job['race'],))

    def export_index(self):
        data=self.snapshot()
        atomic_json(self.root/'index.json',data)

    def snapshot(self):
        with self.store.connect() as db:
            rows=db.execute('SELECT a.id,a.directory,a.summary,r.data FROM race_archives a JOIN races r ON a.id=r.id').fetchall()
            counts=dict(db.execute('SELECT state,COUNT(*) FROM archive_assets GROUP BY state'))
            pending_versions=db.execute("SELECT COUNT(DISTINCT v.id) FROM versions v JOIN results r ON r.version=v.id WHERE v.download_state!='downloaded'").fetchone()[0]
        races=[]
        for row in rows:
            manifest=json.loads(row['summary']);race=json.loads(row['data'])
            races.append({'race':race,'directory':row['directory'],'updated_at':manifest.get('updated_at'),
                          'state':manifest.get('state','indexing'),'participant_count':manifest.get('participant_count',0),
                          'code_counts':manifest.get('code_counts',{}),'file_counts':manifest.get('file_counts',{}),
                          'run_count':manifest.get('run_count',0),'run_lists_indexed':manifest.get('run_lists_indexed',0),
                          'manifest_url':'/race-files/'+row['id']+'/manifest.json'})
        races.sort(key=lambda r:r['race'].get('qualifying_closes_at') or '',reverse=True)
        return {'races':races,'assets':counts,'race_versions_missing_code':pending_versions,
                'health':self.store.meta('race_archive_health',{}),'download_progress':self.store.meta('download_progress',{}),'updated_at':utc()}

    def race_manifest(self, ident):
        with self.store.connect() as db:
            row=db.execute('SELECT manifest FROM race_archives WHERE id=?',(ident,)).fetchone()
        return json.loads(row[0]) if row else None

    def file(self, ident, relative):
        with self.store.connect() as db:
            row=db.execute('SELECT directory FROM race_archives WHERE id=?',(ident,)).fetchone()
        if not row or not relative or '..' in Path(relative).parts:
            return None
        parent=(self.store.root/row[0]).resolve()
        path=(parent/relative).resolve()
        return path if path.is_relative_to(parent) and path.is_file() and not path.name.endswith(('.incoming','.tmp')) else None
