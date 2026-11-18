"""Persistent race priority, historical backfill progress and local notifications."""
from datetime import datetime, timedelta, timezone
import json
import time

from registry import epoch, utc


class DownloadState:
    def __init__(self, store):
        self.store=store
        with store.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS archive_priorities (
                    race TEXT PRIMARY KEY, promoted REAL, completed REAL,
                    scope TEXT, winner TEXT);
                CREATE TABLE IF NOT EXISTS archive_batches (
                    id INTEGER PRIMARY KEY, started TEXT, completed TEXT,
                    generation INTEGER DEFAULT 1);
                CREATE TABLE IF NOT EXISTS archive_batch_races (
                    batch INTEGER, race TEXT, PRIMARY KEY(batch,race));
                CREATE TABLE IF NOT EXISTS archive_notifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, event_key TEXT UNIQUE,
                    created_at TEXT, message TEXT);
                CREATE INDEX IF NOT EXISTS results_version ON results(version,race);
            ''')
        if self.store.meta('priority_tracking_started') is None:
            self.store.set_meta('priority_tracking_started',time.time())

    def ensure_batch(self):
        # Freeze membership only after historical race pagination has finished.
        # Newly concluded races use their own priority queue and do not reset it.
        if not self.store.meta('race_scan',{}).get('completed_at'):
            return
        with self.store.connect() as db:
            if db.execute('SELECT 1 FROM archive_batches WHERE id=1').fetchone():return
            db.execute('INSERT INTO archive_batches(id,started) VALUES (1,?)',(utc(),))
            db.execute('''INSERT INTO archive_batch_races SELECT 1,id FROM races
                WHERE json_extract(data,'$.status') IN ('RACE_COMPLETE','RACE_CANCELLED','CANCELLED')''')

    @staticmethod
    def observe(store, db, race, previous):
        if race.get('status')!='RACE_COMPLETE':return
        completed=epoch(race.get('race_completed_at')) or time.time()
        fresh=(previous is not None and previous.get('status')!='RACE_COMPLETE') or (
            previous is None and completed>=store.meta('priority_tracking_started',time.time()))
        if not fresh:return
        DownloadState.promote(db,race,completed)

    @staticmethod
    def promote(db, race, completed=None):
        completed=completed or epoch(race.get('race_completed_at')) or time.time()
        db.execute('''INSERT INTO archive_priorities VALUES (?,?,?,?,?) ON CONFLICT(race)
            DO UPDATE SET promoted=excluded.promoted,completed=excluded.completed,
            scope='all',winner=excluded.winner''',
            (race['race_id'],time.time(),completed,'all',race.get('winner_agent_version_id')))
        db.execute('UPDATE race_archives SET dirty=dirty+1,materialized_at=0 WHERE id=?',(race['race_id'],))
        db.execute('UPDATE races SET checked=0 WHERE id=?',(race['race_id'],))
        db.execute("UPDATE versions SET next_try=0 WHERE id=? AND download_state!='downloaded'",
                   (race.get('winner_agent_version_id'),))
        db.execute("UPDATE archive_assets SET state='pending',next_try=0 WHERE kind='run_list' AND version IN (SELECT version FROM results WHERE race=?)",(race['race_id'],))
        # The prior race day's code is checked with the API, never assumed public.
        # This promotes the five-day release cohort without refetching its old data.
        if race.get('qualifying_closes_at'):
            target=(datetime.fromtimestamp(epoch(race['qualifying_closes_at']),timezone.utc)-timedelta(days=5)).date().isoformat()
            for row in db.execute('SELECT id,data FROM races WHERE suite=?',(race['suite_id'],)).fetchall():
                earlier=json.loads(row['data'])
                if (earlier.get('qualifying_closes_at') or '')[:10]==target:
                    db.execute('INSERT OR IGNORE INTO archive_priorities VALUES (?,?,?,?,NULL)',
                        (row['id'],time.time(),completed-.001,'code'))

    def bootstrap_priority(self):
        if self.store.meta('priority_bootstrapped'):return
        with self.store.connect() as db:
            recent=[]
            for row in db.execute('SELECT data FROM races'):
                race=json.loads(row[0])
                if race.get('status')=='RACE_COMPLETE' and epoch(race.get('race_completed_at'))>=time.time()-172800:
                    recent.append(race)
            if recent:self.promote(db,max(recent,key=lambda r:epoch(r.get('race_completed_at'))))
        if not recent and not self.store.meta('race_scan',{}).get('completed_at'):return
        self.store.set_meta('priority_bootstrapped',True)

    def source(self):
        with self.store.connect() as db:
            row=db.execute('''SELECT v.*,p.race AS priority_race FROM archive_priorities p JOIN versions v
                ON v.id=p.winner OR EXISTS(SELECT 1 FROM results r WHERE r.race=p.race AND r.version=v.id)
                WHERE v.download_state!='downloaded' AND v.next_try<=?
                ORDER BY p.completed DESC,
                    CASE WHEN v.id=p.winner THEN 0 WHEN v.download_state='waiting_release' THEN 1
                         WHEN p.scope='code' THEN 2 ELSE 3 END,v.id LIMIT 1''',(time.time(),)).fetchone()
        return dict(row) if row else None

    def race_to_refresh(self):
        with self.store.connect() as db:
            row=db.execute("SELECT r.id FROM races r JOIN archive_priorities p ON p.race=r.id WHERE p.scope='all' AND r.checked=0 ORDER BY p.completed DESC LIMIT 1").fetchone()
        return row[0] if row else None

    @staticmethod
    def asset_match(alias='a'):
        return f'''p.scope='all' AND ({alias}.race=p.race OR
            ({alias}.kind='run_list' AND EXISTS(SELECT 1 FROM results r WHERE r.race=p.race AND r.version={alias}.version)) OR
            ({alias}.kind='suite_qualifiers' AND {alias}.suite=(SELECT suite FROM races WHERE id=p.race)))'''

    def asset(self):
        with self.store.connect() as db:
            priorities=db.execute("SELECT p.race,r.suite FROM archive_priorities p JOIN races r ON r.id=p.race WHERE p.scope='all' ORDER BY p.completed DESC").fetchall()
            for priority in priorities:
                row=db.execute('''SELECT * FROM (
                    SELECT * FROM archive_assets WHERE race=? AND state!='downloaded' AND next_try<=?
                    UNION SELECT * FROM archive_assets WHERE kind='run_list' AND version IN
                        (SELECT version FROM results WHERE race=?) AND state!='downloaded' AND next_try<=?
                    UNION SELECT * FROM archive_assets WHERE kind='suite_qualifiers' AND suite=?
                        AND state!='downloaded' AND next_try<=?)
                    ORDER BY CASE kind WHEN 'run_list' THEN 0 WHEN 'run_detail' THEN 1 ELSE 2 END,
                    CASE state WHEN 'pending' THEN 0 ELSE 1 END,checked,id LIMIT 1''',
                    (priority['race'],time.time(),priority['race'],time.time(),priority['suite'],time.time())).fetchone()
                if row:return {**dict(row),'priority_race':priority['race']}
        return None

    def materialization(self):
        with self.store.connect() as db:
            row=db.execute('''SELECT a.*,r.data,r.checked FROM race_archives a JOIN races r ON r.id=a.id
                JOIN archive_priorities p ON p.race=a.id WHERE (a.dirty>0 OR a.synced<r.checked)
                AND a.materialized_at<? ORDER BY p.completed DESC,a.synced LIMIT 1''',(time.time()-15,)).fetchone()
        return dict(row) if row else None

    def priority_summary(self, db):
        source=self.source();asset=self.asset()
        ids={r['priority_race'] for r in (source,asset) if r}
        if not ids:return None
        rows=db.execute('SELECT p.race,p.completed,p.scope,r.data FROM archive_priorities p JOIN races r ON p.race=r.id ORDER BY p.completed DESC').fetchall()
        selected=next(r for r in rows if r['race'] in ids);race=json.loads(selected['data'])
        return {'race_id':race['race_id'],'race_number':race['race_number'],'suite_id':race['suite_id'],
                'scope':selected['scope'],'active':True}

    def update(self):
        self.ensure_batch()
        with self.store.connect() as db:
            batch=db.execute('SELECT * FROM archive_batches WHERE id=1').fetchone()
            priority=self.priority_summary(db)
            if not batch:
                progress={'state':'indexing','percent':None,'total':0,'settled':0,'downloaded':0,
                          'pending':0,'waiting_release':0,'unpublished':0,'errors':0,'priority':priority}
            else:
                # Current run-list refreshes may share a version with historical races.
                # An existing snapshot remains usable if it postdates those races.
                assets=db.execute('''WITH scoped AS (SELECT a.id,a.kind,a.path,a.checked,a.state,a.error,a.version
                    FROM archive_assets a WHERE a.race IN (SELECT race FROM archive_batch_races WHERE batch=1)
                    OR (a.kind='run_list' AND a.version IN (SELECT r.version FROM results r JOIN archive_batch_races b ON r.race=b.race WHERE b.batch=1))
                    OR (a.kind='suite_qualifiers' AND a.suite IN (SELECT r.suite FROM races r JOIN archive_batch_races b ON r.id=b.race WHERE b.batch=1)))
                    SELECT CASE WHEN a.kind='run_list' AND a.path IS NOT NULL AND a.checked>=COALESCE((
                        SELECT MAX(CAST(strftime('%s',json_extract(r.data,'$.race_completed_at')) AS REAL))
                        FROM races r JOIN results x ON x.race=r.id JOIN archive_batch_races b ON b.race=r.id
                        WHERE b.batch=1 AND x.version=a.version),0) THEN 'downloaded' ELSE a.state END AS state,
                        a.error,COUNT(*) AS count FROM scoped a GROUP BY 1,2''').fetchall()
                codes=db.execute('''SELECT v.download_state AS state,v.error,COUNT(DISTINCT v.id) AS count
                    FROM versions v JOIN results r ON v.id=r.version JOIN archive_batch_races b ON r.race=b.race
                    WHERE b.batch=1 GROUP BY v.download_state,v.error''').fetchall()
                metadata=db.execute('''SELECT COUNT(*) AS total,SUM(CASE WHEN a.summary!='{}' AND a.dirty=0
                    AND r.checked>0 AND a.synced>=r.checked
                    AND COALESCE(json_extract(a.summary,'$.file_counts.error'),0)=0
                    AND COALESCE(json_extract(a.summary,'$.code_counts.error'),0)=0 THEN 1 ELSE 0 END) AS done
                    ,SUM(CASE WHEN COALESCE(json_extract(a.summary,'$.file_counts.error'),0)>0 OR
                        COALESCE(json_extract(a.summary,'$.code_counts.error'),0)>0 THEN 1 ELSE 0 END) AS errors
                    FROM archive_batch_races b JOIN races r ON r.id=b.race LEFT JOIN race_archives a ON a.id=r.id WHERE b.batch=1''').fetchone()
                metadata_pending=metadata['total']-int(metadata['done'] or 0)-int(metadata['errors'] or 0)
                counts={'downloaded':int(metadata['done'] or 0),'pending':metadata_pending,
                        'waiting_release':0,'unpublished':0,'errors':int(metadata['errors'] or 0)}
                for row in list(assets)+list(codes):
                    state=row['state']
                    if state=='waiting_release' and row['error'] in {'DIAGNOSTICS_UNAVAILABLE','PARTIAL_RESULT'}:state='pending'
                    counts['errors' if state=='error' else state]+=row['count']
                if batch['completed'] and counts['pending']==metadata_pending and not counts['errors']:
                    # Revalidating the same saved roster must not generate another
                    # completion notice. Newly discovered pending resources still reopen it.
                    counts['pending']-=metadata_pending;counts['downloaded']+=metadata_pending
                total=sum(counts.values());settled=counts['downloaded']+counts['waiting_release']+counts['unpublished']
                complete=not counts['pending'] and not counts['errors']
                progress={**counts,'state':'complete' if complete else 'priority' if priority else 'downloading',
                          'total':total,'settled':settled,'percent':100 if complete else (100*settled/total if total else 0),
                          'races':metadata['total'],'priority':priority,'completed_at':batch['completed']}
                if complete and not batch['completed']:
                    completed=utc();progress['completed_at']=completed
                    db.execute('UPDATE archive_batches SET completed=? WHERE id=1',(completed,))
                    message='Historical backfill complete: all currently available indexed files are archived.'
                    if counts['waiting_release']:message+=f" {counts['waiting_release']} items await public release and remain scheduled."
                    db.execute('INSERT OR IGNORE INTO archive_notifications(event_key,created_at,message) VALUES (?,?,?)',
                               (f"backfill:1:{batch['generation']}",completed,message))
                elif not complete and batch['completed']:
                    db.execute('UPDATE archive_batches SET completed=NULL,generation=generation+1 WHERE id=1')
                    progress['completed_at']=None
            progress['events']=[dict(r) for r in db.execute('SELECT id,created_at,message FROM archive_notifications ORDER BY id DESC LIMIT 20')][::-1]
        self.store.set_meta('download_progress',progress)
        return progress
