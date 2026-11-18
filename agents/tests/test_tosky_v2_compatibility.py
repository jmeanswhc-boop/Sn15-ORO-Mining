"""Run inherited financial, protocol and procedure invariants on the candidate."""
from unittest.mock import patch

import test_chidori_procedures as procedures
import test_chidori_v2_transport as transport
import test_chidori_v4_regression as previous
from test_tosky_v2 import agent, ROOT


class PreservedProcedureTests(procedures.ProcedureStateTests):
    def setUp(self):
        item = patch.object(procedures, 'agent', agent)
        item.start()
        self.addCleanup(item.stop)
        # These fixtures isolate the procedure state machine; source-audit
        # interpretation is exercised separately by the candidate intake tests.
        audit = patch.object(agent.ToskyControls, 'ensure_source_coverage', return_value='')
        audit.start()
        self.addCleanup(audit.stop)
        super().setUp()


class PreservedFinancialTests(previous.V4FinancialTests):
    candidate = agent


class PreservedTransportTests(transport.RealClientTransportTests):
    def setUp(self):
        for key, value in (('v2', agent), ('V2_PATH', ROOT/'agents/Tosky_v2.py')):
            item = patch.object(transport, key, value)
            item.start()
            self.addCleanup(item.stop)
        super().setUp()
