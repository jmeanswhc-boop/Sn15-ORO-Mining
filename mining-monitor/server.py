#!/usr/bin/env python3
"""Local, read-only dashboard for ORO's public mining activity."""
import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
import threading
import time
from registry import Registry, Worker, VERSION
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlsplit
from urllib.request import Request, urlopen

API_BASE = "https://api.oroagents.com/v1/public/"
ROOT = Path(__file__).resolve().parent
UUID = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
ROUTES = re.compile(
    rf"(?:suites(?:/current|/\d+/problems)?|top(?:/history)?|top-miner-payout|"
    rf"leaderboard|validators|evaluations/(?:running|pending)|"
    rf"races/(?:current|history|{UUID}(?:/validator-variance|/seed-commitment)?)|"
    rf"agent-versions/{UUID}(?:/status|/runs|/problems)?|"
    rf"evaluation-runs/{UUID}|episode-results/{UUID}/feedback)"
)


def canonical_path(raw):
    """Only permit documented, public reads with bounded query parameters."""
    parsed = urlsplit(raw)
    if parsed.scheme or parsed.netloc or not ROUTES.fullmatch(parsed.path):
        raise ValueError("Unsupported public endpoint")
    params = parse_qs(parsed.query, keep_blank_values=True, max_num_fields=10)
    for key, values in params.items():
        if len(values) != 1:
            raise ValueError("Duplicate query parameter")
        value = values[0]
        if key in {"limit", "offset", "suite_id"}:
            if not value.isdecimal():
                raise ValueError("Invalid numeric parameter")
            number = int(value)
            if number > 1000000 or (key == "limit" and not 1 <= number <= 100):
                raise ValueError("Query parameter outside supported range")
        elif key == "score_type":
            if value not in {"qualifying", "race"}:
                raise ValueError("Invalid score type")
        elif key == "phase":
            if value not in {"QUALIFYING", "RACE"}:
                raise ValueError("Invalid phase")
        elif key == "race_id":
            if not re.fullmatch(UUID, value):
                raise ValueError("Invalid race identifier")
        elif key == "q":
            if len(value) > 200:
                raise ValueError("Search is too long")
        else:
            raise ValueError("Unsupported query parameter")
    query = urlencode(sorted((key, values[0]) for key, values in params.items()))
    return parsed.path + ("?" + query if query else "")


def timestamp():
    return datetime.now(timezone.utc).isoformat()


class PublicAPI:
    """Share cached responses, coalesce reads, pace traffic and honor backoff."""
    def __init__(self):
        self.cache = OrderedDict()
        self.lock = threading.Lock()
        self.next_request = 0.0
        self.backoff_until = 0.0

    def get(self, path):
        with self.lock:
            now = time.monotonic()
            cached = self.cache.get(path)
            if cached and now < cached['expires']:
                self.cache.move_to_end(path)
                return cached['payload']
            if now < self.backoff_until:
                return self.failure(cached, "ORO requested a pause before the next request", 429,
                                    math.ceil(self.backoff_until - now))
            time.sleep(max(0, self.next_request - now))
            self.next_request = time.monotonic() + .8
            try:
                request = Request(API_BASE + path, headers={
                    'Accept': 'application/json', 'Accept-Encoding': 'gzip',
                    'User-Agent': 'oro-public-mining-monitor/1.0',
                })
                with urlopen(request, timeout=20) as response:
                    raw = response.read(32 * 1024 * 1024 + 1)
                    if len(raw) > 32 * 1024 * 1024:
                        raise ValueError("Response exceeded the dashboard size limit")
                    if response.headers.get('Content-Encoding') == 'gzip':
                        raw = gzip.decompress(raw)
                    data = json.loads(raw)
                payload = {'data': data, 'fetched_at': timestamp(), 'stale': False, 'error': None}
                ttl = 15
                if path == 'suites' or path.endswith('/seed-commitment'):
                    ttl = 60
                if path.endswith('/feedback') and data.get('diagnostics_state') == 'available':
                    ttl = 300
                self.cache[path] = {'payload': payload, 'expires': time.monotonic() + ttl}
                self.cache.move_to_end(path)
                while len(self.cache) > 256:
                    self.cache.popitem(last=False)
                return payload
            except HTTPError as error:
                delay = 15
                if error.code == 429:
                    header = error.headers.get('Retry-After', '30')
                    try:
                        delay = max(1, min(3600, int(header)))
                    except ValueError:
                        delay = 30
                    self.backoff_until = time.monotonic() + delay
                payload = self.failure(cached, f"ORO API returned HTTP {error.code}", error.code, delay)
            except (URLError, TimeoutError, OSError, ValueError) as error:
                payload = self.failure(cached, f"Public data could not be refreshed ({type(error).__name__})", 502, 15)
            self.cache[path] = {'payload': payload, 'expires': time.monotonic() + payload['retry_after']}
            while len(self.cache) > 256:
                self.cache.popitem(last=False)
            return payload

    def invalidate(self,path):
        """A race transition requires fresh public metadata before indexing runs."""
        with self.lock:self.cache.pop(path,None)

    def artifact_url(self, ident, artifact_type='AGENT_CODE', eval_run_id=None, problem_id=None):
        """Only request a released public source URL; no miner credentials."""
        if not VERSION.fullmatch(ident):
            raise ValueError('Invalid agent version')
        if artifact_type not in {'AGENT_CODE','EVAL_LOGS_BUNDLE','EVAL_PROBLEM_LOGS','REASONING_DETAILS'}:
            raise ValueError('Unsupported public artifact type')
        for value in (eval_run_id, problem_id):
            if value is not None and not VERSION.fullmatch(value):
                raise ValueError('Invalid public artifact identity')
        with self.lock:
            now = time.monotonic()
            if now < self.backoff_until:
                error = RuntimeError('Public API backoff')
                error.retry_after = math.ceil(self.backoff_until - now)
                raise error
            time.sleep(max(0, self.next_request - now))
            self.next_request = time.monotonic() + .8
            body={'artifact_type':artifact_type,'agent_version_id':ident}
            if eval_run_id:body['eval_run_id']=eval_run_id
            if problem_id:body['problem_id']=problem_id
            request = Request(API_BASE + 'artifacts/download-url',
                              data=json.dumps(body).encode(),
                              headers={'Content-Type': 'application/json', 'Accept': 'application/json',
                                       'User-Agent': 'oro-public-mining-monitor/2.0'})
            try:
                with urlopen(request, timeout=20) as response:
                    return json.loads(response.read(1024 * 1024))
            except HTTPError as response:
                try:
                    detail = json.loads(response.read(65536))
                except (ValueError, OSError):
                    detail = {}
                error = RuntimeError(f'Public artifact HTTP {response.code}')
                # Error envelopes have used both root and nested detail fields.
                error.public_code=detail.get('error_code') if isinstance(detail,dict) else None
                error.release_gated = any(code in json.dumps(detail) for code in ('ARTIFACT_NOT_RELEASED','RACE_IN_FLIGHT'))
                error.not_published = error.public_code=='ARTIFACT_NOT_FOUND'
                try:
                    delay = min(3600, max(1, int(response.headers.get('Retry-After', '300'))))
                except ValueError:
                    delay = 300
                error.retry_after = delay
                if response.code == 429:
                    self.backoff_until = time.monotonic() + delay
                raise error from None

    @staticmethod
    def failure(cached, message, status, delay):
        previous = cached['payload'] if cached else {}
        return {'data': previous.get('data'), 'fetched_at': previous.get('fetched_at'),
                'stale': True, 'error': message, 'status': status, 'retry_after': delay}


class Handler(BaseHTTPRequestHandler):
    api = PublicAPI()
    registry = None
    worker = None

    def do_GET(self):
        if self.headers.get('Host') not in {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}:
            self.send_error(403, 'Local access only')
            return
        path = urlsplit(self.path).path
        if path == '/monitor/races' and self.registry:
            self.send_json(self.race_archive().snapshot())
            return
        if path.startswith('/monitor/races/') and self.registry:
            ident=path.removeprefix('/monitor/races/')
            manifest=self.race_archive().race_manifest(ident) if VERSION.fullmatch(ident) else None
            self.send_json(manifest if manifest else {'error':'Race archive not found'},200 if manifest else 404)
            return
        if path.startswith('/race-files/') and self.registry:
            from urllib.parse import unquote
            parts=path.removeprefix('/race-files/').split('/',1)
            file=self.race_archive().file(parts[0],unquote(parts[1])) if len(parts)==2 and VERSION.fullmatch(parts[0]) else None
            if not file:
                self.send_error(404,'Race file not archived');return
            try:
                from os import fstat
                from urllib.parse import quote
                import shutil
                with file.open('rb') as stream:
                    self.send_response(200)
                    self.send_header('Content-Type','application/octet-stream')
                    self.send_header('Content-Length',str(fstat(stream.fileno()).st_size))
                    self.send_header('Content-Disposition',"attachment; filename*=UTF-8''"+quote(file.name,safe=''))
                    self.send_header('X-Content-Type-Options','nosniff')
                    self.send_header('Cache-Control','no-store')
                    self.end_headers()
                    shutil.copyfileobj(stream,self.wfile,1024*1024)
            except FileNotFoundError:self.send_error(404,'Race file not archived')
            except (BrokenPipeError,ConnectionResetError):pass
            return
        if path == '/monitor/state' and self.registry:
            try:
                query = parse_qs(urlsplit(self.path).query, max_num_fields=5, keep_blank_values=True)
                if set(query) - {'view', 'offset', 'q', 'sort_by', 'order'} or any(len(v) != 1 for v in query.values()):
                    raise ValueError('Unsupported monitor query')
                category = query.get('view', ['champions'])[0]
                offset = int(query.get('offset', ['0'])[0])
                if category not in {'champions', 'tosky', 'archive'} or not 0 <= offset <= 1000000:
                    raise ValueError('Unsupported monitor query')
                self.send_json(self.registry.snapshot(category, offset, query=query.get('q', [''])[0],
                               sort_by=query.get('sort_by', ['champion_since'])[0], order=query.get('order', ['desc'])[0]))
            except ValueError as error:
                self.send_json({'error': str(error)}, 400)
            return
        if path.startswith('/code/') and self.registry:
            ident = path.removeprefix('/code/')
            source = self.registry.archive_file(ident) if VERSION.fullmatch(ident) else None
            if source is None:
                self.send_error(404, 'Released code is not archived yet')
                return
            self.respond(source.read_bytes(), 'application/zip' if source.suffix == '.zip' else 'text/plain; charset=utf-8',
                         filename=ident + source.suffix)
            return
        if path.startswith('/api/'):
            try:
                endpoint = canonical_path(self.path[len('/api/'):])
            except ValueError as error:
                self.send_json({'error': str(error)}, 400)
                return
            self.send_json(self.api.get(endpoint))
            return
        files = {'/': 'index.html', '/app.js': 'app.js', '/styles.css': 'styles.css','/theme.js':'theme.js',
                 '/favicon.svg': 'favicon.svg', '/favicon.ico': 'favicon.ico'}
        if path not in files:
            self.send_error(404)
            return
        content = (ROOT / files[path]).read_bytes()
        mime = {'/': 'text/html; charset=utf-8', '/app.js': 'text/javascript; charset=utf-8',
                '/styles.css': 'text/css; charset=utf-8', '/theme.js':'text/javascript; charset=utf-8',
                '/favicon.svg': 'image/svg+xml', '/favicon.ico': 'image/x-icon'}[path]
        self.respond(content, mime)

    def send_json(self, value, status=200):
        self.respond(json.dumps(value).encode(), 'application/json; charset=utf-8', status)

    def race_archive(self):
        if self.worker:
            return self.worker.race_archive
        from race_archive import RaceArchive
        return RaceArchive(self.registry,self.api)

    def respond(self, content, mime, status=200, filename=None):
        self.send_response(status)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(content)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        if filename:
            self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'")
        self.end_headers()
        try:
            self.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format, *args):
        if len(args) > 1 and str(args[1]) != '200':
            super().log_message(format, *args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--hotkey', help='Fixed public Tosky hotkey; persists across restarts')
    parser.add_argument('--name', default='Tosky', help='Display name; identity always uses the hotkey')
    parser.add_argument('--interval', type=int, default=30, help='Fast monitoring interval, seconds (minimum 15)')
    parser.add_argument('--chain-url', default='https://entrypoint-finney.opentensor.ai:443', help='Read-only Finney endpoint for coldkey ownership')
    parser.add_argument('--no-worker', action='store_true', help='View persisted data without upstream requests')
    parser.add_argument('--root', type=Path, default=ROOT.parent, help='Project directory containing champions/ and challengers/')
    args = parser.parse_args()
    Handler.registry = Registry(args.root, args.hotkey or '', args.name)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    Handler.registry.set_meta('health', {'state': 'read-only' if args.no_worker else 'starting',
                                       'updated_at': timestamp(), 'error': None})
    if not args.no_worker:
        Handler.worker = Worker(Handler.registry, Handler.api, args.interval, args.chain_url)
        Handler.worker.start()
    print(f'ORO mining monitor is running at http://127.0.0.1:{server.server_port}', flush=True)
    print('Press Ctrl+C to stop.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if Handler.worker:
            Handler.worker.stop.set()
            Handler.worker.race_archive.stop.set()
            Handler.worker.thread.join(timeout=30)
            if Handler.worker.race_archive.thread:
                Handler.worker.race_archive.thread.join(timeout=30)
            Handler.registry.set_meta('health', {'state': 'stopped', 'updated_at': timestamp(), 'error': None})
        server.server_close()


if __name__ == '__main__':
    main()
