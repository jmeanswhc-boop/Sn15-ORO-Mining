"""Champion challenge targets from public scores; no qualifying-score substitution."""
from decimal import Decimal, InvalidOperation


def number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        result = Decimal(str(value))
        return result if result.is_finite() else None
    except (InvalidOperation, ValueError):
        return None


def target(top, race, champion, observed_at=None):
    top, race, champion = top or {}, race or {}, champion or {}
    result = {'state': 'unavailable', 'race_id': race.get('race_id'),
              'race_number': race.get('race_number'), 'suite_id': race.get('suite_id'),
              'champion_id': top.get('top_agent_version_id'),
              'champion_name': champion.get('agent_name'), 'observed_at': observed_at,
              'strict': True}
    if not result['champion_id'] or not race.get('race_id') or race.get('suite_id') is None or race.get('suite_id') != top.get('suite_id'):
        result['reason'] = 'Current race and champion are not available in the same suite.'
        return result
    score, threshold, margin = (number(top.get(k)) for k in ('top_score', 'challenge_threshold', 'margin'))
    if threshold is None and score is not None and margin is not None:
        threshold = score + margin
    if score is None or threshold is None or threshold < score:
        result['reason'] = 'Waiting for the public champion score and challenge threshold.'
        return result
    # The backend's threshold is authoritative, including its decaying margin.
    result.update(state='available', champion_overall=str(score), threshold=str(threshold),
                  margin=str(threshold-score), source='public/top')
    return result


def history_delta(item, previous):
    windows = [item.get('window') or []] + [r['result'].get('window') or [] for r in item.get('races', [])]
    total = Decimal(0)
    for race in previous:
        entry = next((w for window in windows for w in window if w.get('race_id') == race['race_id']
                      and w.get('suite_id') == race['suite_id']), None)
        if entry is not None:
            delta = number(entry.get('delta'))
            if delta is None:
                return None
            total += delta
        else:
            # Only a version submitted after that race closed is provably new.
            # Missing history for an older submission is never treated as zero.
            from registry import epoch
            try:
                seeded = epoch(item.get('submitted_at')) > epoch(race.get('qualifying_closes_at')) > 0
            except (ValueError, TypeError):
                seeded = False
            if not seeded:
                return None
    return total


def requirement(item, challenge, race, champion, previous):
    result = {'state': 'unavailable', 'threshold': challenge.get('threshold'), 'strict': True}
    if challenge['state'] != 'available':
        result['reason'] = challenge.get('reason')
        return result
    if item['agent_version_id'] == challenge['champion_id']:
        result['state'] = 'incumbent'
        return result
    entry = next((r for r in item.get('races', []) if r['race']['race_id'] == race['race_id']), None)
    if entry is None:
        result.update(state='not_participating', reason='Not in the latest published roster for this race.')
        return result
    if item.get('eliminated_at') or entry['result'].get('eliminated_at') or item.get('is_discarded'):
        result.update(state='ineligible', reason='Eliminated or discarded; cannot challenge in this race.')
        return result
    # Use only this race's Overall. A prior-race Overall or public qualifier is
    # not a live score, even when the version status still exposes it.
    overall = number(entry['result'].get('weighted_score'))
    threshold = number(challenge['threshold'])
    result.update(state='pending' if overall is None else 'above_threshold' if overall > threshold else 'below_threshold',
                  overall=str(overall) if overall is not None else None)
    if overall is not None:
        result['gap'] = str(threshold-overall)

    if race.get('status') not in {'RACE_RUNNING', 'QUALIFYING_CLOSED', 'QUALIFYING_OPEN'}:
        return result
    # Planning estimate: O = current anchor + (current delta + retained deltas)/3.
    # Subtracting two Overalls cancels the common anchor. Therefore the needed
    # raw lead is 3*margin + champion's retained deltas - challenger's deltas.
    # Require the actual two preceding race records; do not guess launch windows.
    if len(previous) != 2 or not champion:
        return result
    own_delta, champion_delta = history_delta(item, previous), history_delta(champion, previous)
    if own_delta is None or champion_delta is None:
        return result
    lead = 3*number(challenge['margin']) + champion_delta-own_delta
    result.update(raw_lead_estimate=str(lead), retained_delta=str(own_delta),
                  champion_retained_delta=str(champion_delta), window_size=3)
    champion_entry = next((r for r in champion.get('races', []) if r['race']['race_id'] == race['race_id']), None)
    raw = number(champion_entry['result'].get('race_score')) if champion_entry else None
    if raw is not None:
        result['raw_target_estimate'] = str(raw+lead)
    return result
