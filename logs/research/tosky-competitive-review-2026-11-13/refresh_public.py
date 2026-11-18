"""Small, resumable public metadata refresh. No credentials or inference calls."""
import json
from datetime import datetime, timezone
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent
BASE = 'https://api.oroagents.com/v1/public/'
HOTKEY = '5CmLJBFGahsEcYamtvyY4zKZXdTejqMAQMZ9HjbQG2dfE4Xh'
ENDPOINTS = {
    'top': BASE+'top',
    'current-race': BASE+'races/current',
    'current-suite': BASE+'suites/current',
    'owned-versions': BASE+'leaderboard?q='+HOTKEY+'&limit=100',
    'models': BASE+'inference/models?provider=openrouter',
    'public-qualifiers': BASE+'suites/160/problems',
    'known-tosky-status': BASE+'agent-versions/06af5d7e-bfd4-761c-8000-575e592cb36b/status',
    'haiku-pricing': 'https://openrouter.ai/api/v1/models/anthropic/claude-haiku-5.5/endpoints',
    'sonnet-pricing': 'https://openrouter.ai/api/v1/models/anthropic/claude-sonnet-5.5/endpoints',
    'deepseek-pricing': 'https://openrouter.ai/api/v1/models/deepseek/deepseek-v3.2/endpoints',
}

def main():
    for name,url in ENDPOINTS.items():
        path=ROOT/(name+'.json')
        if path.exists() and json.loads(path.read_text()).get('http_status')==200:
            continue
        envelope={'url':url,'fetched_at':datetime.now(timezone.utc).isoformat()}
        try:
            response=requests.get(url,timeout=20)
            envelope.update(http_status=response.status_code,data=response.json())
        except (requests.RequestException,ValueError) as exc:
            envelope.update(error=type(exc).__name__)
        path.write_text(json.dumps(envelope,indent=2)+'\n')
        print(name,envelope.get('http_status'),envelope.get('error'),flush=True)

if __name__=='__main__':main()
