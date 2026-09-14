import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from traffic_policy import TrafficPolicy


def stamp(value):
    return datetime.fromisoformat(value).replace(tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()


class TrafficPolicyTests(unittest.TestCase):
    def setUp(self):
        self.now = stamp('2026-09-14T22:29:59')
        self.policy = TrafficPolicy('fixture-protected', 'fixture-background', clock=lambda: self.now)

    def test_only_trusted_credentials_classify(self):
        self.assertEqual(self.policy.authenticate('Bearer '+'fixture-protected'), 'protected')
        self.assertEqual(self.policy.authenticate('Bearer '+'fixture-background'), 'background')
        for value in (None, '', 'Bearer '+'background', 'background', 'Bearer '+'fixture-protected X-Workload: background'):
            self.assertIsNone(self.policy.authenticate(value))
        self.assertEqual(self.policy.admit(None, 'POST', '/v1/chat/completions'), 401)

    def test_default_existing_key_protected(self):
        p = TrafficPolicy('fixture-protected')
        self.assertEqual(p.authenticate('Bearer '+'fixture-protected'), 'protected')
        self.assertIsNone(p.authenticate('Bearer '+'fixture-background'))

    def test_equal_keys_rejected(self):
        with self.assertRaises(ValueError):
            TrafficPolicy('same', 'same')

    def test_cross_midnight_boundaries(self):
        for value, expected in [('22:29:59', False), ('22:30:00', True), ('23:59:59', True),
                                ('00:00:00', True), ('07:29:59', True), ('07:30:00', False)]:
            with self.subTest(value=value):
                self.assertEqual(self.policy.in_night_window(stamp('2026-09-14T'+value)), expected)

    def test_day_window(self):
        p = TrafficPolicy('fixture', night_start='09:00', night_end='17:00')
        for hour, expected in [('08:59:59', False), ('09:00:00', True), ('16:59:59', True), ('17:00:00', False), ('00:00:00', False)]:
            self.assertEqual(p.in_night_window(stamp('2026-09-14T'+hour)), expected)

    def test_reject_background_without_resetting_idle(self):
        before = self.policy.snapshot()
        self.now = stamp('2026-09-14T23:00:00')
        self.assertEqual(self.policy.admit('background', 'POST', '/v1/chat/completions'), 503)
        self.assertEqual(self.policy.admit('protected', 'POST', '/v1/chat/completions'), 200)
        after = self.policy.snapshot()
        self.assertEqual(before['last_protected_at'], after['last_protected_at'])
        self.assertEqual(before['last_background_at'], after['last_background_at'])
        self.assertEqual(after['background_rejected_total'], 1)

    def test_health_and_reads_not_business(self):
        for method, path in [('GET', '/health'), ('POST', '/health?probe=1'), ('GET', '/v1/models')]:
            self.assertFalse(self.policy.is_business(method, path))
            self.assertIsNone(self.policy.begin('background', method, path))
        self.assertEqual(self.policy.snapshot()['background_inflight'], 0)

    def test_inflight_and_completion_time(self):
        a = self.policy.begin('protected', 'POST', '/v1/chat/completions')
        b = self.policy.begin('background', 'POST', '/v1/responses')
        self.assertEqual(self.policy.snapshot()['protected_inflight'], 1)
        self.now += 600
        self.policy.end(b)
        self.assertEqual(self.policy.snapshot()['background_inflight'], 0)
        self.assertEqual(self.policy.snapshot()['last_background_at'], self.now)
        self.assertEqual(self.policy.snapshot()['protected_inflight'], 1)
        self.policy.end(a)
        self.assertEqual(self.policy.snapshot()['last_protected_at'], self.now)
        with self.assertRaises(ValueError):
            self.policy.end(a)

    def test_invalid_windows(self):
        for start, end in [('24:00','07:30'), ('22:60','07:30'), ('1:00','07:30'), ('22:30','22:30')]:
            with self.assertRaises(ValueError):
                TrafficPolicy('fixture', night_start=start, night_end=end)


if __name__ == '__main__':
    unittest.main()

class ForcedDeadlineTests(unittest.TestCase):
    def test_disabled_by_default(self):
        self.assertFalse(TrafficPolicy('fixture').force_allowed(stamp('2026-09-14T23:45:00')))

    def test_cross_midnight_force_boundaries(self):
        p=TrafficPolicy('fixture',force_training_at='23:30')
        for clock,expected in [('22:29:59',False),('22:30:00',False),('23:29:59',False),
                               ('23:30:00',True),('23:59:59',True),('00:00:00',True),
                               ('07:29:59',True),('07:30:00',False),('12:00:00',False)]:
            with self.subTest(clock=clock):
                self.assertEqual(p.force_allowed(stamp('2026-09-14T'+clock)),expected)

    def test_invalid_deadlines_fail_closed(self):
        for value in ['22:29','07:30','12:00','24:00','invalid',True]:
            with self.subTest(value=value),self.assertRaises(ValueError):
                TrafficPolicy('fixture',force_training_at=value)
