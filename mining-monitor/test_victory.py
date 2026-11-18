"""Offline score-to-beat checks: strict boundaries, live scope, and seeded history."""
from decimal import Decimal
import tempfile
import unittest

from registry import Registry
from victory import target, requirement
from test_registry import A, B, HOTKEY


class VictoryTests(unittest.TestCase):
    def setUp(self):
        self.race={'race_id':'current','suite_id':160,'race_number':8,'status':'RACE_RUNNING'}
        self.previous=[{'race_id':'p7','suite_id':160,'race_number':7,'status':'RACE_COMPLETE',
                        'qualifying_closes_at':'2020-01-07T19:00:00'},
                       {'race_id':'p6','suite_id':160,'race_number':6,'status':'RACE_COMPLETE',
                        'qualifying_closes_at':'2020-01-06T19:00:00'}]
        self.top={'suite_id':160,'top_agent_version_id':A,'top_score':.3,
                  'margin':.01,'challenge_threshold':.31}
        self.champion={'agent_version_id':A,'agent_name':'Champion','window':[
            {'race_id':'p7','suite_id':160,'delta':.04},
            {'race_id':'p6','suite_id':160,'delta':.02}]}
        self.item={'agent_version_id':B,'submitted_at':'2020-01-08T10:00:00',
                   'final_score':.95,'weighted_score':.9,
                   'races':[{'race':self.race,'result':{'weighted_score':None,'race_score':None}}]}
        self.challenge=target(self.top,self.race,self.champion)

    def calc(self):
        return requirement(self.item,self.challenge,self.race,self.champion,self.previous)

    def test_authoritative_threshold_includes_margin_once(self):
        # A policy/rounded margin must never override the published threshold.
        top=dict(self.top,challenge_threshold=.312345678901,margin=.02)
        c=target(top,self.race,self.champion)
        self.assertEqual(Decimal(c['threshold']),Decimal('.312345678901'))
        self.assertEqual(Decimal(c['margin']),Decimal('.012345678901'))

    def test_additive_fallback_and_missing_or_nonfinite_data(self):
        top=dict(self.top);del top['challenge_threshold']
        self.assertEqual(target(top,self.race,self.champion)['threshold'],'0.31')
        for value in [None,float('nan'),float('inf'),True]:
            self.assertEqual(target(dict(self.top,top_score=value),self.race,self.champion)['state'],'unavailable')
        self.assertEqual(target(self.top,None,self.champion)['state'],'unavailable')
        self.assertEqual(target(self.top,dict(self.race,suite_id=3),self.champion)['state'],'unavailable')

    def test_equality_does_not_win_but_one_small_increment_does(self):
        self.item['races'][0]['result']['weighted_score']=.31
        self.assertEqual(self.calc()['state'],'below_threshold')
        self.assertEqual(Decimal(self.calc()['gap']),0)
        self.item['races'][0]['result']['weighted_score']=.310000000001
        self.assertEqual(self.calc()['state'],'above_threshold')

    def test_current_gap_uses_overall_and_zero_is_a_real_score(self):
        self.item['races'][0]['result']['weighted_score']=.2
        self.assertEqual(Decimal(self.calc()['gap']),Decimal('.11'))
        self.item['races'][0]['result']['weighted_score']=0
        self.assertEqual(Decimal(self.calc()['gap']),Decimal('.31'))

    def test_withheld_does_not_use_qualifier_or_previous_overall(self):
        result=self.calc()
        self.assertEqual(result['state'],'pending');self.assertIsNone(result['overall'])
        self.assertNotIn('gap',result)

    def test_new_submission_needs_history_lead_plus_three_margins(self):
        self.assertEqual(Decimal(self.calc()['raw_lead_estimate']),Decimal('.09'))
        # Common field baseline cancels: even +.08 raw still loses on Overall.
        champion_raw=Decimal('.4');baseline=Decimal('.35')
        champion_overall=baseline+(champion_raw-baseline+Decimal('.06'))/3
        challenger_overall=baseline+(champion_raw+Decimal('.08')-baseline)/3
        self.assertLess(challenger_overall,champion_overall+Decimal('.01'))

    def test_old_submission_missing_history_is_not_zero_seeded(self):
        self.item['submitted_at']='2020-01-01T10:00:00'
        self.assertNotIn('raw_lead_estimate',self.calc())

    def test_existing_history_can_reduce_required_raw_lead(self):
        self.item['window']=[{'race_id':'p7','suite_id':160,'delta':.08},
                             {'race_id':'p6','suite_id':160,'delta':.03}]
        self.assertEqual(Decimal(self.calc()['raw_lead_estimate']),Decimal('-.02'))

    def test_absolute_raw_target_only_after_champion_raw_is_released(self):
        self.assertNotIn('raw_target_estimate',self.calc())
        self.champion['races']=[{'race':self.race,'result':{'race_score':.4}}]
        self.assertEqual(Decimal(self.calc()['raw_target_estimate']),Decimal('.49'))

    def test_eliminated_nonparticipant_and_incumbent_not_claimed_as_winners(self):
        self.item['eliminated_at']='2020-01-08'
        self.assertEqual(self.calc()['state'],'ineligible')
        self.item['races']=[]
        self.assertEqual(self.calc()['state'],'not_participating')
        self.item['agent_version_id']=A
        self.assertEqual(self.calc()['state'],'incumbent')

    def test_no_three_race_prediction_for_launch_or_settled_race(self):
        self.previous=self.previous[:1]
        self.assertNotIn('raw_lead_estimate',self.calc())
        self.previous+= [dict(self.previous[0],race_id='p6')]
        self.race['status']='RACE_COMPLETE'
        self.assertNotIn('raw_lead_estimate',self.calc())

    def test_registry_snapshot_attaches_only_current_batch_comparison(self):
        with tempfile.TemporaryDirectory() as path:
            store=Registry(path,HOTKEY);store.set_meta('suites',[{'suite_id':160}])
            store.set_meta('top',self.top);store.set_meta('current_race',self.race)
            store.upsert(self.champion,160,champion=True)
            store.upsert({**self.item,'miner_hotkey':HOTKEY},160)
            for race in self.previous:store.record_race({'race':race,'qualifiers':[]})
            store.record_race({'race':self.race,'qualifiers':[
                {'agent_version_id':B,'weighted_score':None,'qualifying_score':.95},
                {'agent_version_id':A,'weighted_score':None,'race_score':None}]})
            snap=store.snapshot('tosky');self.assertEqual(snap['total'],1)
            self.assertEqual(snap['challenge']['threshold'],'0.31')
            r=snap['entries'][0]['win_requirement']
            self.assertEqual(r['state'],'pending');self.assertEqual(Decimal(r['raw_lead_estimate']),Decimal('.09'))
            store.set_meta('current_race',dict(self.race,race_id='tomorrow',race_number=9))
            self.assertEqual(store.snapshot('tosky')['entries'][0]['win_requirement']['state'],'not_participating')


if __name__=='__main__':unittest.main()
