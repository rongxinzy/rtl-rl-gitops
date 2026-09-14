import datetime as dt
import unittest
import controller


def now(clock, day=14):
    return dt.datetime.fromisoformat(f'2026-09-{day:02d}T{clock}:00+08:00')


class FakeEffects:
    def __init__(self, phase='glm_ready', **route):
        self.host_state = {'phase': phase, 'job_id': 'job', 'operator_enabled': True}
        self.route = dict(active_backend='primary', converged=True,
                          primary_inflight=3, business_idle_seconds=0, **{})
        self.route.update(route)
        self.calls = []
        self.backup = False
    def host(self, action, deadline=None):
        self.calls.append(('host', action, deadline))
        return dict(self.host_state)
    def router(self):
        return dict(self.route)
    def switch(self, backend, force=False):
        self.calls.append(('switch', backend, force))
    def probe(self, backend):
        self.calls.append(('probe', backend))
        return self.backup if backend == 'backup' else True
    def mutations(self):
        return [c for c in self.calls if c[0] == 'switch' or c[:2] in [('host', 'training'), ('host', 'inference')]]


def obj(**spec):
    return {'metadata': {'generation': 1}, 'spec': dict(
        timezone='Asia/Shanghai', trainingStart='22:30', trainingStop='07:30',
        forceTrainingAt='23:30', mode='Auto', controlMode='Active',
        allowNightWithoutBackup=True, jobId='job', **spec)}


class DeadlineTests(unittest.TestCase):
    def test_boundaries(self):
        for clock, day, desired, forced in [
            ('22:29',14,'Inference',False),('22:30',14,'Training',False),
            ('23:29',14,'Training',False),('23:30',14,'Training',True),
            ('07:29',15,'Training',True),('07:30',15,'Inference',False)]:
            with self.subTest(clock=clock):
                self.assertEqual(controller.desired(obj()['spec'], now(clock,day)), desired)
                self.assertEqual(controller.force_training_due(obj()['spec'],now(clock,day)),forced)
    def test_missing_and_manual(self):
        spec=obj()['spec'];spec.pop('forceTrainingAt')
        self.assertFalse(controller.force_training_due(spec,now('23:30')))
        for mode in ['Inference','Training']:
            spec.update(mode=mode,forceTrainingAt='23:30',trainingUntil='2026-09-15T00:00:00Z')
            self.assertFalse(controller.force_training_due(spec,now('23:30')))
    def test_invalid_cutoff(self):
        for cutoff in ['07:30','22:29','25:00']:
            spec=obj()['spec'];spec['forceTrainingAt']=cutoff
            with self.assertRaises(ValueError):controller.force_training_due(spec,now('23:30'))
    def test_day_window_and_timezone(self):
        spec=obj()['spec'];spec.update(trainingStart='09:00',trainingStop='17:00',forceTrainingAt='10:00')
        self.assertTrue(controller.force_training_due(spec,now('10:00').astimezone(dt.timezone.utc)))
        self.assertFalse(controller.force_training_due(spec,now('09:59')))


class ReconcileTests(unittest.TestCase):
    def run_at(self, clock, effects=None, changes=None, day=14):
        effects=effects or FakeEffects()
        value=obj();value['spec'].update(changes or {})
        return controller.reconcile(value,effects,now(clock,day)),effects
    def test_busy_before_deadline(self):
        for clock in ['22:30','23:29']:
            result,e=self.run_at(clock)
            self.assertEqual(result['phase'],'WaitingForBusinessIdle')
            self.assertEqual(e.mutations(),[])
    def test_idle_before_deadline(self):
        e=FakeEffects(primary_inflight=0,business_idle_seconds=300)
        result,e=self.run_at('22:30',e)
        self.assertEqual(e.mutations(),[('switch','maintenance',False)])
        e=FakeEffects(active_backend='maintenance',primary_inflight=0,business_idle_seconds=300)
        self.run_at('22:30',e)
        self.assertEqual(e.mutations()[0][:2],('host','training'))
    def test_deadline_maintenance_first(self):
        result,e=self.run_at('23:30')
        self.assertEqual(result['phase'],'ForcingMaintenance')
        self.assertEqual(e.mutations(),[('switch','maintenance',True)])
        self.assertFalse(any(c[0]=='probe' for c in e.calls))
    def test_maintenance_unconverged_wait(self):
        result,e=self.run_at('23:30',FakeEffects(active_backend='maintenance',converged=False))
        self.assertEqual(result['phase'],'ForcingMaintenance')
        self.assertEqual(e.mutations(),[])
    def test_busy_backup_down_force_training(self):
        result,e=self.run_at('23:30',FakeEffects(active_backend='maintenance'))
        self.assertEqual(e.mutations()[0][:2],('host','training'))
        self.assertTrue(result['forceTraining'])
        self.assertFalse(any(c[0]=='probe' for c in e.calls))
    def test_job_mismatch(self):
        e=FakeEffects();e.host_state['job_id']='other'
        result,e=self.run_at('23:30',e)
        self.assertEqual(result['phase'],'BlockedJobMismatch');self.assertEqual(e.mutations(),[])
    def test_suspended_observe(self):
        for changes in [{'suspend':True},{'controlMode':'Observe'}]:
            _,e=self.run_at('23:30',changes=changes)
            self.assertEqual(e.mutations(),[])
    def test_too_late_gate_retained(self):
        result,e=self.run_at('07:29',FakeEffects(active_backend='maintenance'),day=15)
        self.assertEqual(result['phase'],'TooLateToStartTraining')
        self.assertNotIn(('host','training'),[c[:2] for c in e.calls])
    def test_already_committed_ignores_idle_and_inflight(self):
        for phase in ['training','stopping_glm','starting_training']:
            for clock in ['23:29','23:30']:
                result,e=self.run_at(clock,FakeEffects(phase,active_backend='maintenance'))
                self.assertEqual(e.mutations()[0][:2],('host','training'))
                self.assertFalse(any(c[:2]==('switch','primary') for c in e.calls))
    def test_durable_stop_marker_survives_host_status_override(self):
        for clock, day in [('23:29',14),('00:01',15)]:
            e=FakeEffects(active_backend='maintenance')
            e.host_state.update(glm_stop_window='2026-09-14',glm_stop_started=now('22:40').timestamp())
            self.run_at(clock,e,day=day)
            self.assertEqual(e.mutations()[0][:2],('host','training'))
        e=FakeEffects(active_backend='maintenance')
        e.host_state.update(glm_stop_window='2026-09-13',glm_stop_started=now('22:40',13).timestamp())
        result,e=self.run_at('23:29',e)
        self.assertEqual(result['phase'],'WaitingForBusinessIdle')
        self.assertEqual(e.mutations(),[('switch','primary',False)])
    def test_restart_reobserves_route(self):
        for _ in range(2):
            _,e=self.run_at('23:30')
            self.assertEqual(e.mutations(),[('switch','maintenance',True)])
        _,e=self.run_at('23:30',FakeEffects(active_backend='maintenance'))
        self.assertEqual(e.mutations()[0][:2],('host','training'))
    def test_old_policy_without_force(self):
        value=obj();value['spec'].pop('forceTrainingAt')
        e=FakeEffects()
        result=controller.reconcile(value,e,now('23:30'))
        self.assertEqual(result['phase'],'WaitingForBusinessIdle');self.assertEqual(e.mutations(),[])
    def test_morning_restores_inference(self):
        _,e=self.run_at('07:30',FakeEffects('training',active_backend='maintenance'),day=15)
        self.assertTrue(any(c[:2]==('host','inference') for c in e.calls))
        self.assertFalse(any(c[:2]==('host','training') for c in e.calls))

if __name__=='__main__':unittest.main()
