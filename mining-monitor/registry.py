"""Durable public OROBench inventory. Downloaded agents are never imported."""
from __future__ import annotations

import ast
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import hashlib
from functools import partial
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import threading
import time
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen
import zipfile

VERSION = re.compile(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")
MAX_SOURCE = 64 * 1024 * 1024
CHAMPION_SORTS = {'champion_since', 'overall_score', 'qualifier_score', 'hotkey', 'coldkey'}


def utc():
    return datetime.now(timezone.utc).isoformat()


def epoch(value):
    if not value:
        return 0
    date = datetime.fromisoformat(value.replace('Z', '+00:00'))
    return date.replace(tzinfo=date.tzinfo or timezone.utc).timestamp()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def public_qualifier(item, races):
    value = item.get('final_score')
    if value is None:
        value = item.get('latest_final_score')
    if value is not None:
        return {'score': value, 'suite_id': item.get('suite_id'), 'source': 'version'}
    for entry in races:
        value = entry['result'].get('qualifying_score')
        if value is not None:
            return {'score': value, 'suite_id': entry['race']['suite_id'],
                    'race_id': entry['race'].get('race_id'), 'source': 'historical_race_roster'}
    return {'score': None, 'suite_id': item.get('suite_id'), 'source': 'unpublished'}


def champion_fields(item, top):
    titles = item.get('titles', [])
    dates = []
    for title in titles:
        try:
            if title.get('top_at'):
                dates.append((epoch(title['top_at']), title['top_at']))
        except (ValueError, TypeError):
            continue
    item['champion_since'] = min(dates)[1] if dates else None
    scores = [t['final_score'] for t in titles if t.get('final_score') is not None]
    if top and item['agent_version_id'] == top.get('top_agent_version_id'):
        item['champion_score'] = top.get('top_score')
    else:
        item['champion_score'] = scores[-1] if scores else item.get('weighted_score')


def search_sort_champions(records, query, sort_by, order):
    query = query.strip().casefold()
    records = [item for item in records if not query or any(
        query in (item.get(field) or '').casefold() for field in ('agent_name', 'miner_hotkey', 'coldkey'))]

    def value(item):
        if sort_by == 'overall_score':
            return item.get('champion_score')
        if sort_by == 'qualifier_score':
            return item['public_qualifier']['score']
        if sort_by == 'champion_since':
            return epoch(item['champion_since']) if item.get('champion_since') else None
        return item.get('miner_hotkey' if sort_by == 'hotkey' else 'coldkey') or None

    # Unknown scores/keys/dates stay last in either direction; zero is a real score.
    known = sorted((item for item in records if value(item) is not None), key=lambda i: i['agent_version_id'])
    unknown = sorted((item for item in records if value(item) is None), key=lambda i: i['agent_version_id'])
    return sorted(known, key=value, reverse=order == 'desc') + unknown


def unpack_source(raw, destination):
    """Validate the complete archive before writing; impose extraction bounds."""
    destination.mkdir(parents=True, exist_ok=True)
    members = []
    if zipfile.is_zipfile(io.BytesIO(raw)):
        with zipfile.ZipFile(io.BytesIO(raw)) as bundle:
            infos = bundle.infolist()
            if len(infos) > 2000 or sum(i.file_size for i in infos) > 128 * 1024 * 1024:
                raise ValueError('Source archive exceeds extraction limits')
            names = set()
            for info in infos:
                path = PurePosixPath(info.filename)
                if (path.is_absolute() or '..' in path.parts or '\\' in info.filename
                        or not path.parts or ':' in path.parts[0]
                        or (info.external_attr >> 16) & 0o170000 == 0o120000
                        or info.filename in names):
                    raise ValueError('Unsafe source archive member')
                names.add(info.filename)
            for info in infos:
                if info.is_dir():
                    continue
                target = destination.joinpath(*PurePosixPath(info.filename).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(bundle.read(info))
                members.append(info.filename)
        return 'agent_code.zip', members
    # Public archival preserves the exact released bytes, including old submissions
    # whose syntax is unsupported by this interpreter. Never execute downloaded code.
    (destination / 'agent.py').write_bytes(raw)
    return 'agent.py', ['agent.py']


class Registry:
    def __init__(self, root, hotkey='', name='Tosky'):
        self.root = Path(root).resolve()
        self.data_dir = self.root / 'logs' / 'orobench-monitor'
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db = self.data_dir / 'registry.sqlite3'
        with self.connect() as db:
            db.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, data TEXT);
                CREATE TABLE IF NOT EXISTS versions (
                    id TEXT PRIMARY KEY, suite INTEGER, data TEXT NOT NULL,
                    champion INTEGER DEFAULT 0, title TEXT DEFAULT '[]',
                    archive TEXT, sha256 TEXT, download_state TEXT DEFAULT 'pending',
                    next_try REAL DEFAULT 0, error TEXT, checked REAL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS races (
                    id TEXT PRIMARY KEY, suite INTEGER, data TEXT, checked REAL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS results (
                    race TEXT, version TEXT, data TEXT, PRIMARY KEY (race, version));
            ''')
        config = self.meta('config', {})
        if hotkey:
            if not re.fullmatch(r'[1-9A-HJ-NP-Za-km-z]{47,49}', hotkey):
                raise ValueError('Invalid tracking hotkey')
            config.update(hotkey=hotkey, name=name)
        elif not config:
            config = {'name': name, 'hotkey': ''}
        self.set_meta('config', config)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.db, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def meta(self, key, default=None):
        with self.connect() as db:
            row = db.execute('SELECT data FROM meta WHERE key=?', (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_meta(self, key, value):
        with self.connect() as db:
            db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)', (key, json.dumps(value)))

    def upsert(self, value, suite=None, champion=False, title=None, connection=None):
        ident = value.get('agent_version_id')
        if not ident or not VERSION.fullmatch(ident):
            return
        with (nullcontext(connection) if connection is not None else self.connect()) as db:
            old = db.execute('SELECT * FROM versions WHERE id=?', (ident,)).fetchone()
            data = json.loads(old['data']) if old else {}
            data.update(value)
            suite = suite if suite is not None else value.get('suite_id', old['suite'] if old else None)
            titles = json.loads(old['title']) if old else []
            if title:
                previous = next((t for t in titles if t.get('top_at') == title.get('top_at')
                                 and t.get('suite_id') == title.get('suite_id')), None)
                if previous is not None:
                    previous.update({k: v for k, v in title.items() if v is not None})
                else:
                    titles.append(title)
                titles.sort(key=lambda t: t.get('top_at') or '')
            db.execute('''INSERT INTO versions (id,suite,data,champion,title) VALUES (?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET suite=excluded.suite,data=excluded.data,
                champion=MAX(versions.champion,excluded.champion),title=excluded.title''',
                (ident, suite, json.dumps(data), int(champion), json.dumps(titles)))
            if champion and (not old or not old['champion']):
                db.execute("UPDATE versions SET next_try=0 WHERE id=? AND download_state!='downloaded'",(ident,))

    def version(self, ident):
        with self.connect() as db:
            row = db.execute('SELECT * FROM versions WHERE id=?', (ident,)).fetchone()
        return dict(row) if row else None

    def import_champions(self):
        suites = {s['suite_id'] for s in self.meta('suites', [])}
        count = 0
        for path in (self.root / 'champions').glob('*/metadata.json'):
            try:
                metadata = json.loads(path.read_text())
                detail = metadata['api_detail']
                if detail.get('suite_id') not in suites:
                    continue
                # Existing race-winner archives are not automatically titleholders.
                self.upsert(detail)
                artifact = path.parent / metadata.get('artifact_file', 'agent.py')
                if artifact.parent != path.parent or not artifact.is_file():
                    continue
                digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
                if digest != metadata.get('archive_sha256'):
                    continue
                with self.connect() as db:
                    db.execute("UPDATE versions SET archive=?,sha256=?,download_state='downloaded' WHERE id=?",
                               (str(artifact.relative_to(self.root)), digest, detail['agent_version_id']))
                count += 1
            except (OSError, ValueError, KeyError):
                continue
        self.set_meta('existing_archives_verified', count)

    def record_race(self, payload):
        race = payload.get('race')
        allowed = {s['suite_id'] for s in self.meta('suites', [])}
        if not race or race.get('suite_id') not in allowed:
            return
        with self.connect() as db:
            previous=db.execute('SELECT data FROM races WHERE id=?',(race['race_id'],)).fetchone()
            db.execute('INSERT OR REPLACE INTO races VALUES (?,?,?,?)',
                       (race['race_id'], race['suite_id'], json.dumps(race), time.time()))
            if 'qualifiers' in payload:
                db.execute('DELETE FROM results WHERE race=?',(race['race_id'],))
            for item in payload.get('qualifiers', []):
                db.execute('INSERT OR REPLACE INTO results VALUES (?,?,?)',
                           (race['race_id'], item['agent_version_id'], json.dumps(item)))
                # Per-race scores/status stay in results; old races must not overwrite current state.
                self.upsert({k: item[k] for k in ('agent_version_id', 'agent_name', 'miner_hotkey', 'version_number') if k in item}, race['suite_id'], connection=db)
            if db.execute("SELECT 1 FROM sqlite_master WHERE name='race_archives'").fetchone():
                db.execute('UPDATE race_archives SET dirty=dirty+1 WHERE id=?',(race['race_id'],))
                if previous and json.loads(previous[0]).get('status')!=race.get('status'):
                    db.execute("UPDATE archive_assets SET state='pending',next_try=0 WHERE kind='run_list' AND version IN (SELECT version FROM results WHERE race=?)",(race['race_id'],))
            if db.execute("SELECT 1 FROM sqlite_master WHERE name='archive_priorities'").fetchone():
                from download_state import DownloadState
                DownloadState.observe(self,db,race,json.loads(previous[0]) if previous else None)
        winner = race.get('winner_agent_version_id')
        if winner:
            self.upsert({'agent_version_id': winner, 'agent_name': race.get('winner_agent_name')}, race['suite_id'])

    def import_race_snapshots(self):
        """Seed public scores already on disk; preserve fresher live observations."""
        paths = sorted((self.root / 'problems' / 'races').glob('*/race_detail.json'))
        cursor = self.meta('race_snapshot_cursor', 0)
        for path in paths[cursor:cursor + 1]:
            try:
                payload = json.loads(path.read_text())
                ident = payload['race']['race_id']
                with self.connect() as db:
                    old = db.execute('SELECT checked FROM races WHERE id=?', (ident,)).fetchone()
                if old and old[0] > 0:
                    continue
                self.record_race(payload)
                with self.connect() as db:
                    db.execute('UPDATE races SET checked=0 WHERE id=?', (ident,))
            except (OSError, ValueError, KeyError):
                continue
        self.set_meta('race_snapshot_cursor', cursor + 1)
        if cursor + 1 >= len(paths):
            self.set_meta('race_snapshots_imported', True)

    def archive_file(self, ident):
        row = self.version(ident)
        if not row or not row['archive'] or row['download_state'] != 'downloaded':
            return None
        path = (self.root / row['archive']).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            return None
        return path

    def export_catalog(self):
        """Keep a complete titleholder index alongside the existing source folders."""
        with self.connect() as db:
            rows = db.execute('SELECT * FROM versions WHERE champion=1 ORDER BY id').fetchall()
        records = []
        for row in rows:
            value = json.loads(row['data'])
            with self.connect() as db:
                result_rows = db.execute('''SELECT results.data AS result,races.data AS race FROM results
                    JOIN races ON results.race=races.id WHERE version=?
                    ORDER BY json_extract(races.data,'$.qualifying_closes_at') DESC''', (row['id'],)).fetchall()
            races = [{'race': json.loads(r['race']), 'result': json.loads(r['result'])} for r in result_rows]
            records.append({'agent_version_id': row['id'], 'agent_name': value.get('agent_name'),
                            'version_number': value.get('version_number'), 'suite_id': row['suite'],
                            'submitted_at': value.get('submitted_at'), 'miner_hotkey': value.get('miner_hotkey'),
                            'public_qualifier_score': value.get('final_score', value.get('latest_final_score')),
                            'public_qualifier': public_qualifier(dict(value, suite_id=row['suite']), races),
                            'titles': json.loads(row['title']), 'archive': row['archive'],
                            'sha256': row['sha256'], 'download_state': row['download_state']})
        atomic_json(self.root / 'champions' / 'registry.json', {'updated_at': utc(), 'champions': records})

    def promote_archives(self):
        with self.connect() as db:
            rows = db.execute("SELECT * FROM versions WHERE champion=1 AND archive LIKE 'challengers/%'").fetchall()
        for row in rows:
            artifact = self.archive_file(row['id'])
            if not artifact:
                continue
            destination = self.root / 'champions' / artifact.parent.name
            if destination.exists():
                target = destination / artifact.name
                if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != row['sha256']:
                    continue
            else:
                shutil.copytree(artifact.parent, destination)
            with self.connect() as db:
                db.execute('UPDATE versions SET archive=? WHERE id=?',
                           (str((destination / artifact.name).relative_to(self.root)), row['id']))

    def snapshot(self, category='champions', offset=0, limit=100, query='', sort_by='champion_since', order='desc'):
        if len(query) > 200 or sort_by not in CHAMPION_SORTS or order not in {'asc', 'desc'}:
            raise ValueError('Unsupported champion search or sort')
        config = self.meta('config', {})
        top = self.meta('top')
        current_race = self.meta('current_race')
        with self.connect() as db:
            rows = db.execute('SELECT * FROM versions ORDER BY champion DESC,id DESC').fetchall()
            counts = dict(db.execute('SELECT download_state,COUNT(*) FROM versions GROUP BY download_state'))
            champions = sum(r['champion'] for r in rows)
            downloaded_champions = sum(bool(r['champion']) and r['download_state'] == 'downloaded' for r in rows)
            owned = [r for r in rows if config.get('hotkey') and json.loads(r['data']).get('miner_hotkey') == config['hotkey']]
            selected = owned if category == 'tosky' else [r for r in rows if r['champion']] if category == 'champions' else rows
            race_counts = db.execute('SELECT COUNT(*),SUM(checked>0) FROM races').fetchone()
            total = len(selected)
            records = []
            for row in (selected if category == 'champions' else selected[offset:offset + limit]):
                item = json.loads(row['data'])
                item.update(agent_version_id=row['id'], suite_id=row['suite'], champion=bool(row['champion']),
                            titles=json.loads(row['title']), archive_state=row['download_state'],
                            archive_error=row['error'], archive_sha256=row['sha256'],
                            code_url='/code/' + row['id'] if row['archive'] and row['download_state'] == 'downloaded' else None)
                result_rows = db.execute('''SELECT results.data AS result,races.data AS race FROM results
                    JOIN races ON results.race=races.id WHERE version=?
                    ORDER BY json_extract(races.data,'$.qualifying_closes_at') DESC,
                             races.suite DESC,json_extract(races.data,'$.race_number') DESC''', (row['id'],)).fetchall()
                item['races'] = [{'race': json.loads(r['race']), 'result': json.loads(r['result'])} for r in result_rows]
                item['public_qualifier'] = public_qualifier(item, item['races'])
                owner = self.meta('owner:' + (item.get('miner_hotkey') or ''), {})
                item['coldkey'] = owner.get('coldkey')
                item['coldkey_provenance'] = owner
                if category == 'champions':
                    champion_fields(item, top)
                records.append(item)
            if category == 'champions':
                records = search_sort_champions(records, query, sort_by, order)
                total = len(records)
                records = records[offset:offset + limit]
            challenge = None
            if category == 'tosky':
                from victory import target, requirement
                champion_row = db.execute('SELECT data FROM versions WHERE id=?',
                                          ((top or {}).get('top_agent_version_id'),)).fetchone()
                champion = json.loads(champion_row[0]) if champion_row else {}
                champion['races'] = [{'race': json.loads(r['race']), 'result': json.loads(r['result'])}
                    for r in db.execute('''SELECT races.data AS race,results.data AS result FROM results
                        JOIN races ON races.id=results.race WHERE version=?
                        ORDER BY json_extract(races.data,'$.qualifying_closes_at') DESC''',
                        ((top or {}).get('top_agent_version_id'),))]
                challenge = target(top, current_race, champion, self.meta('top_checked_at'))
                previous = [json.loads(r[0]) for r in db.execute('''SELECT data FROM races WHERE suite=?
                    AND json_extract(data,'$.status')='RACE_COMPLETE'
                    AND json_extract(data,'$.race_number')<?
                    ORDER BY json_extract(data,'$.race_number') DESC LIMIT 2''',
                    ((current_race or {}).get('suite_id'), (current_race or {}).get('race_number')))]
                for item in records:
                    item['win_requirement'] = requirement(item, challenge, current_race or {}, champion, previous)
        return {'config': config, 'entries': records, 'total': total, 'offset': offset,
                'summary': {'versions': len(rows), 'champions': champions, 'champions_downloaded': downloaded_champions,
                            'tosky_versions': len(owned), 'downloads': counts,
                            'races_discovered': race_counts[0], 'races_refreshed': race_counts[1] or 0},
                'top': top, 'current_race': current_race, 'challenge': challenge,
                'suites': self.meta('suites', []), 'health': self.meta('health', {}),
                'fast_refresh': self.meta('fast_refresh'), 'backfill': self.meta('backfill', {}),
                'chain': self.meta('chain', {}), 'download_progress':self.meta('download_progress',{}), 'generated_at': utc()}


class Worker:
    """One paced collector: fast monitoring preempts each small backfill step."""
    def __init__(self, registry, api, interval=30, chain_url=None):
        self.store, self.api = registry, api
        self.interval = max(15, interval)
        self.chain_url = chain_url
        self.stop = threading.Event()
        self.wake = threading.Event()
        self.thread = None
        self.owner_thread = None
        self.chain = None
        self.last_fast = 0
        self.last_history = 0
        from race_archive import RaceArchive
        self.race_archive=RaceArchive(registry,api)

    def get(self, path):
        result = self.api.get(path)
        if result.get('error') or result.get('data') is None or result.get('stale'):
            raise RuntimeError(result.get('error') or 'Public response unavailable')
        return result['data']

    def start(self):
        self.thread = threading.Thread(target=self.run, name='orobench-archive', daemon=True)
        self.thread.start()
        self.owner_thread = threading.Thread(target=self.run_owners, name='orobench-owners', daemon=True)
        self.owner_thread.start()
        self.race_archive.start()

    def run_owners(self):
        # Chain availability must never delay champion detection or source downloads.
        while not self.stop.is_set():
            self.owner_step()
            self.stop.wait(5 if self.chain is not None else 60)

    def run(self):
        while not self.stop.is_set():
            try:
                if time.monotonic() - self.last_fast >= self.interval or self.wake.is_set():
                    self.wake.clear()
                    self.fast()
                    self.last_fast = time.monotonic()
                self.step()
                self.store.set_meta('health', {'state': 'running', 'updated_at': utc(), 'error': None})
            except Exception as error:
                # Do not persist signed storage URLs or remote response bodies in errors.
                self.store.set_meta('health', {'state': 'retrying', 'updated_at': utc(),
                                             'error': str(error) if isinstance(error, RuntimeError) else type(error).__name__})
                self.stop.wait(15)
            self.stop.wait(.1)

    def fast(self):
        if not self.store.meta('suites') or time.time() - self.store.meta('suite_checked', 0) > 3600:
            suites = self.get('suites')
            self.store.set_meta('suites', [s for s in suites if s.get('execution_kind') == 'envpack'])
            self.store.set_meta('suite_checked', time.time())
            if not self.store.meta('imported'):
                self.store.import_champions()
                self.store.set_meta('imported', True)
        suite_ids = {s['suite_id'] for s in self.store.meta('suites', [])}
        top = self.get('top')
        self.store.set_meta('top', top if top.get('suite_id') in suite_ids else None)
        self.store.set_meta('top_checked_at', utc())
        history = self.get('top/history')
        for item in history.get('entries', []):
            if item.get('suite_id') in suite_ids:
                title = dict(item)
                title['observed_at'] = None  # Stable identity across repeated polls.
                value = {k: item[k] for k in ('agent_version_id','agent_name','miner_hotkey','version_number','suite_id') if k in item}
                self.store.upsert(value, champion=True, title=title)
        if top.get('suite_id') in suite_ids and top.get('top_agent_version_id'):
            self.store.upsert({'agent_version_id': top['top_agent_version_id'],
                               'miner_hotkey': top.get('top_miner_hotkey')}, top['suite_id'], champion=True)
        self.store.promote_archives()
        current = self.get('races/current')
        self.store.record_race(current)
        self.store.set_meta('current_race', current.get('race') if (current.get('race') or {}).get('suite_id') in suite_ids else None)
        # Completion discovery must continue even while priority files keep the
        # historical cursor paused. The current endpoint may already show tomorrow.
        self.race_page()
        for endpoint in ('evaluations/running', 'evaluations/pending'):
            evaluations = self.get(endpoint)
            items = evaluations if isinstance(evaluations, list) else evaluations.get('items', [])
            for item in items:
                if item.get('suite_id') in suite_ids:
                    self.store.upsert({k: item[k] for k in ('agent_version_id', 'agent_name', 'miner_hotkey',
                                      'version_number', 'suite_id', 'submitted_at') if k in item})
        config = self.store.meta('config', {})
        if config.get('hotkey'):
            # Name searches are discovery only; ownership always uses the fixed hotkey.
            page = self.get('leaderboard?' + urlencode({'q': config['hotkey'], 'limit': 100}))
            for value in page.get('entries', []):
                if value.get('miner_hotkey') == config['hotkey'] and page['suite']['suite_id'] in suite_ids:
                    self.store.upsert({k: v for k, v in value.items() if k != 'rank'}, page['suite']['suite_id'])
        with self.store.connect() as db:
            tracked = [r['id'] for r in db.execute('SELECT id,data FROM versions').fetchall()
                       if config.get('hotkey') and json.loads(r['data']).get('miner_hotkey') == config['hotkey']]
        champion = top.get('top_agent_version_id') if top.get('suite_id') in suite_ids else None
        for ident in dict.fromkeys(([champion] if champion else []) + tracked):
            try:
                self.enrich(ident, runs=ident in tracked)
            except RuntimeError:
                continue
        if champion:
            row = self.store.version(champion)
            if row and row['download_state'] != 'downloaded' and row['next_try'] <= time.time():
                self.download(row)
        self.store.set_meta('fast_refresh', utc())
        self.store.export_catalog()

    def enrich(self, ident, runs=False):
        status = self.get('agent-versions/' + ident + '/status')
        status['status_checked_at'] = utc()
        self.store.upsert(status)
        if runs:
            self.store.upsert({'agent_version_id': ident, 'runs': self.get('agent-versions/' + ident + '/runs')})
        with self.store.connect() as db:
            db.execute('UPDATE versions SET checked=? WHERE id=?', (time.time(), ident))
        row = self.store.version(ident)
        if (row and not row['champion'] and row['download_state'] != 'downloaded'
                and status.get('release_state')!='RELEASED'
                and epoch(status.get('code_available_at')) > time.time()):
            self.defer(ident, 'waiting_release', min(epoch(status['code_available_at']), time.time() + 3600), None)
        return status

    def step(self):
        """Round-robin work is persisted at page/file boundaries, not held in RAM."""
        if not self.store.meta('race_snapshots_imported'):
            self.store.import_race_snapshots()
        fresh_race=self.race_archive.scheduling.race_to_refresh()
        if fresh_race:
            path='races/'+fresh_race
            if hasattr(self.api,'invalidate'):self.api.invalidate(path)
            self.store.record_race(self.get(path));return
        urgent=self.race_archive.scheduling.source()
        if urgent:
            self.download(urgent);return
        if self.race_archive.scheduling.asset():
            self.stop.wait(.5);return
        turn = self.store.meta('turn', 0)
        self.store.set_meta('turn', turn + 1)
        # Champion backfill precedes the bulk challenger queue.
        with self.store.connect() as db:
            champion = db.execute("SELECT * FROM versions WHERE champion=1 AND download_state!='downloaded' AND next_try<=? ORDER BY id DESC LIMIT 1", (time.time(),)).fetchone()
        if champion:
            self.download(dict(champion))
            return
        with self.store.connect() as db:
            champion = db.execute('SELECT id FROM versions WHERE champion=1 AND checked<? LIMIT 1',
                                  (time.time() - 86400,)).fetchone()
        if champion:
            try:
                self.enrich(champion['id'])
            except RuntimeError:
                # Schedule a later retry without blocking every other work item.
                with self.store.connect() as db:
                    db.execute('UPDATE versions SET checked=? WHERE id=?', (time.time() - 86400 + 300, champion['id']))
            return
        action = turn % 5
        if action == 0:
            self.inventory_page()
        elif action == 1:
            self.race_page()
        elif action == 2:
            self.race_detail()
        elif action == 3:
            with self.store.connect() as db:
                row = db.execute("""SELECT * FROM versions WHERE download_state!='downloaded' AND next_try<=?
                    ORDER BY CASE WHEN EXISTS(SELECT 1 FROM results r WHERE r.version=versions.id) THEN 0 ELSE 1 END,
                    CASE download_state WHEN 'waiting_release' THEN 0 WHEN 'pending' THEN 1 ELSE 2 END,
                    next_try,id LIMIT 1""", (time.time(),)).fetchone()
            if row:
                self.download(dict(row))

    def inventory_page(self):
        state = self.store.meta('backfill', {})
        scans = state.setdefault('leaderboards', {})
        suites = self.store.meta('suites', [])
        active = next((s['suite_id'] for s in suites if s.get('is_active')), None)
        jobs = [(s['suite_id'], 'qualifying') for s in suites]
        if active:
            jobs.insert(1, (active, 'race'))
        for suite, score_type in jobs:
            key = str(suite) + ':' + score_type
            scan = scans.setdefault(key, {'offset': 0, 'passes': 0, 'next_scan': 0})
            if scan.get('next_scan', 0) > time.time():
                continue
            page = self.get('leaderboard?' + urlencode({'suite_id': suite, 'score_type': score_type,
                            'limit': 100, 'offset': scan['offset']}))
            batch = page['entries']
            if not batch and scan['offset'] < page['total']:
                raise RuntimeError('Leaderboard pagination ended before the reported total')
            with self.store.connect() as db:
                for item in batch:
                    value = {k: v for k, v in item.items() if k != 'rank'}
                    rank_type = 'global_race' if score_type == 'race' else 'qualifying'
                    value[rank_type + '_rank'] = item.get('rank')
                    value[rank_type + '_rank_at'] = utc()
                    value[rank_type + '_rank_suite'] = suite
                    was_top = bool(item.get('was_top') or item.get('is_current_top'))
                    title = {'top_at': item.get('top_at'), 'suite_id': suite,
                             'final_score': None, 'source': 'leaderboard'} if was_top else None
                    self.store.upsert(value, suite, champion=was_top, title=title, connection=db)
            scan.update(offset=scan['offset'] + len(batch), total=page['total'], updated_at=utc())
            if scan['offset'] >= page['total']:
                scan.update(offset=0, passes=scan['passes'] + 1, completed_at=utc(),
                            next_scan=time.time() + (300 if suite == active else 21600))
            self.store.set_meta('backfill', state)
            return

    def race_page(self):
        state = self.store.meta('race_scan', {'offset': 0, 'next_scan': 0})
        if state['next_scan'] > time.time():
            return
        page = self.get(f"races/history?limit=100&offset={state['offset']}")
        allowed = {s['suite_id'] for s in self.store.meta('suites', [])}
        batch = page['races']
        if not batch and state['offset'] < page['total']:
            raise RuntimeError('Race pagination ended before the reported total')
        with self.store.connect() as db:
            for race in batch:
                if race.get('suite_id') in allowed:
                    previous=db.execute('SELECT data FROM races WHERE id=?',(race['race_id'],)).fetchone()
                    from download_state import DownloadState
                    DownloadState.observe(self.store,db,race,json.loads(previous[0]) if previous else None)
                    db.execute('''INSERT INTO races (id,suite,data) VALUES (?,?,?)
                        ON CONFLICT(id) DO UPDATE SET data=excluded.data''',
                               (race['race_id'], race['suite_id'], json.dumps(race)))
        state.update(offset=state['offset'] + len(batch), total=page['total'])
        if state['offset'] >= page['total']:
            state.update(offset=0, next_scan=time.time() + 300, completed_at=utc())
        self.store.set_meta('race_scan', state)

    def race_detail(self):
        with self.store.connect() as db:
            row = db.execute('SELECT * FROM races WHERE checked<? ORDER BY checked,id DESC LIMIT 1', (time.time() - 3600,)).fetchone()
        if row:
            self.store.record_race(self.get('races/' + row['id']))

    def download(self, row):
        ident = row['id']
        try:
            status = self.enrich(ident)
            available = epoch(status.get('code_available_at'))
            if available > time.time() and status.get('release_state') != 'RELEASED' and not row['champion']:
                self.defer(ident, 'waiting_release', min(available, time.time() + 3600), None)
                return
            artifact = self.api.artifact_url(ident)
            url = artifact['download_url']
            parsed = urlsplit(url)
            if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password:
                raise ValueError('Unsupported artifact URL')
            with urlopen(Request(url, headers={'Accept-Encoding': 'identity'}), timeout=25) as response:
                raw = response.read(MAX_SOURCE + 1)
                encoding = response.headers.get('Content-Encoding')
                expected=response.headers.get('Content-Length')
                if expected is not None and len(raw)!=int(expected):
                    raise EOFError('Public source transfer ended before Content-Length')
            if len(raw) > MAX_SOURCE:
                raise ValueError('Artifact size limit exceeded')
            if encoding == 'gzip':
                import gzip
                with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
                    raw = stream.read(MAX_SOURCE + 1)
                if len(raw) > MAX_SOURCE:
                    raise ValueError('Decoded artifact size limit exceeded')
            parent = self.store.root / ('champions' if row['champion'] else 'challengers')
            slug = re.sub(r'[^A-Za-z0-9._-]', '-', status.get('agent_name') or 'unnamed')[:80].strip('.')
            directory = parent / f"{slug}_v{status.get('version_number', 0)}_{ident}"
            temporary = parent / ('.incoming-' + ident)
            if temporary.exists():
                shutil.rmtree(temporary)
            temporary.mkdir(parents=True)
            artifact_file, members = unpack_source(raw, temporary / 'source')
            (temporary / artifact_file).write_bytes(raw)
            digest = hashlib.sha256(raw).hexdigest()
            metadata = {'api_detail': status, 'archive_sha256': digest, 'archive_bytes': len(raw),
                        'artifact_file': artifact_file, 'collected_at': utc(), 'members': members,
                        'provenance': {'detail': 'https://api.oroagents.com/v1/public/agent-versions/' + ident + '/status',
                                       'artifact': 'https://api.oroagents.com/v1/public/artifacts/download-url',
                                       'presigned_url_redacted': True}}
            if artifact_file=='agent.py':
                try:ast.parse(raw);metadata['syntax_valid']=True
                except (SyntaxError,ValueError):metadata['syntax_valid']=False
            atomic_json(temporary / 'metadata.json', metadata)
            if directory.exists():
                # Never overwrite an existing archive with different bytes.
                existing = directory / artifact_file
                if not existing.exists() or hashlib.sha256(existing.read_bytes()).hexdigest() != digest:
                    raise ValueError('Existing archive conflicts with downloaded source')
                shutil.rmtree(temporary)
            else:
                temporary.replace(directory)
            with self.store.connect() as db:
                db.execute("UPDATE versions SET archive=?,sha256=?,download_state='downloaded',error=NULL,next_try=0 WHERE id=?",
                           (str((directory / artifact_file).relative_to(self.store.root)), digest, ident))
            if row['champion']:
                self.store.export_catalog()
            self.race_archive.mark_dirty(ident)
        except Exception as error:
            gated = getattr(error, 'release_gated', False)
            delay = max(30, getattr(error, 'retry_after', 300))
            self.defer(ident, 'waiting_release' if gated else 'error', time.time() + delay,
                       'ARTIFACT_NOT_RELEASED' if gated else type(error).__name__)

    def defer(self, ident, state, next_try, error):
        with self.store.connect() as db:
            db.execute('UPDATE versions SET download_state=?,next_try=?,error=? WHERE id=?', (state, next_try, error, ident))
        self.race_archive.mark_dirty(ident)

    def owner_step(self):
        if not self.chain_url:
            self.store.set_meta('chain', {'state': 'disabled', 'error': 'No chain endpoint configured'})
            return
        try:
            if self.chain is None:
                from substrateinterface import SubstrateInterface
                self.chain = SubstrateInterface(url=self.chain_url, ss58_format=42,
                                                ws_options={'timeout': 10})
                session = getattr(self.chain.transport, 'session', None)
                if session is not None:
                    session.request = partial(session.request, timeout=(5, 15))
            with self.store.connect() as db:
                rows = db.execute('SELECT data FROM versions WHERE champion=1').fetchall()
            keys = {json.loads(r[0]).get('miner_hotkey') for r in rows}
            keys.add(self.store.meta('config', {}).get('hotkey'))
            for hotkey in sorted(k for k in keys if k):
                previous = self.store.meta('owner:' + hotkey, {})
                if epoch(previous.get('checked_at')) > time.time() - 86400:
                    continue
                block = self.chain.get_chain_finalised_head()
                result = self.chain.query('SubtensorModule', 'Owner', [hotkey], block_hash=block).value
                self.store.set_meta('owner:' + hotkey, {'coldkey': result, 'checked_at': utc(),
                                    'block_hash': block, 'source': 'SubtensorModule.Owner', 'endpoint': self.chain_url})
                break
            self.store.set_meta('chain', {'state': 'connected', 'checked_at': utc()})
        except Exception as error:
            self.chain = None
            self.store.set_meta('chain', {'state': 'unavailable', 'error': type(error).__name__, 'checked_at': utc()})
