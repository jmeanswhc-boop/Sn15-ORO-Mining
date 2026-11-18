"""Regression checks for public routing, caching, and outage handling."""
import gzip
import io
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError
import server


class Response(io.BytesIO):
    def __init__(self, data, compressed=False):
        raw = json.dumps(data).encode()
        super().__init__(gzip.compress(raw) if compressed else raw)
        self.headers = {'Content-Encoding': 'gzip'} if compressed else {}


class RouteTests(unittest.TestCase):
    def test_public_route_and_query_canonicalization(self):
        self.assertEqual(server.canonical_path('leaderboard?q=a%26b&offset=0&limit=50'),
                         'leaderboard?limit=50&offset=0&q=a%26b')

    def test_arbitrary_destinations_and_private_routes_are_rejected(self):
        for route in ['https://example.com', '//example.com', '../health', 'miner/agents',
                      'leaderboard?limit=999', 'leaderboard?limit=1&limit=2',
                      'leaderboard?secret=x', 'leaderboard?score_type=overall', 'evaluations/pending?offset=-1']:
            with self.subTest(route=route), self.assertRaises(ValueError):
                server.canonical_path(route)

    def test_released_feedback_route(self):
        route = 'episode-results/12345678-1234-1234-1234-123456789abc/feedback'
        self.assertEqual(server.canonical_path(route), route)


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.api = server.PublicAPI()
        self.sleep = patch('server.time.sleep').start()
        self.addCleanup(patch.stopall)

    @patch('server.urlopen')
    def test_gzip_decoding_and_coalesced_cache(self, request):
        request.return_value = Response({'race': None}, compressed=True)
        first = self.api.get('races/current')
        self.assertEqual(first['data'], {'race': None})
        self.assertFalse(first['stale'])
        self.assertEqual(first, self.api.get('races/current'))
        self.assertEqual(request.call_count, 1)

    @patch('server.urlopen')
    def test_outage_retains_original_timestamp_and_data(self, request):
        request.return_value = Response({'total': 0})
        initial = self.api.get('leaderboard')
        self.api.cache['leaderboard']['expires'] = 0
        request.side_effect = URLError('unreachable')
        stale = self.api.get('leaderboard')
        self.assertTrue(stale['stale'])
        self.assertEqual(stale['fetched_at'], initial['fetched_at'])
        self.assertEqual(stale['data'], {'total': 0})
        self.assertIsNotNone(stale['error'])

    @patch('server.urlopen')
    def test_rate_limit_applies_to_other_endpoints(self, request):
        request.side_effect = HTTPError('url', 429, 'limited', {'Retry-After': '75'}, None)
        result = self.api.get('validators')
        self.assertEqual(result['retry_after'], 75)
        self.assertEqual(self.api.get('top')['status'], 429)
        self.assertEqual(request.call_count, 1)

    @patch('server.urlopen')
    def test_initial_failure_has_no_fake_data(self, request):
        request.side_effect = URLError('unreachable')
        result = self.api.get('top')
        self.assertIsNone(result['data'])
        self.assertIsNone(result['fetched_at'])

    @patch('server.urlopen')
    def test_unavailable_feedback_is_short_lived(self, request):
        request.return_value = Response({'diagnostics_state': 'diagnostics_unavailable'})
        self.api.get('episode-results/example/feedback')
        entry = self.api.cache['episode-results/example/feedback']
        self.assertLessEqual(entry['expires'] - server.time.monotonic(), 15)

    @patch('server.urlopen')
    def test_cache_recovers_after_failure(self, request):
        request.side_effect = URLError('unreachable')
        self.api.get('top')
        self.api.cache['top']['expires'] = 0
        request.side_effect = None
        request.return_value = Response({'top_score': 0})
        result = self.api.get('top')
        self.assertFalse(result['stale'])
        self.assertEqual(result['data']['top_score'], 0)


if __name__ == '__main__':
    unittest.main()
