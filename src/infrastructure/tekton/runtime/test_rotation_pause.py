import unittest
from unittest.mock import patch
import task
import bridge

JOB='l20-'+'a'*24
UID='11111111-1111-4111-8111-111111111111'

class Clock:
 def __init__(self):self.now=0
 def monotonic(self):return self.now
 def sleep(self,seconds):self.now+=seconds

class PauseTests(unittest.TestCase):
 def state(self,paused=False,complete=False):
  return {'job_id':JOB,'run_uid':UID,'orchestrator':'tekton','job_sha256':'b'*64,'phase':'training','rotation_paused':paused,'verified_complete':{'training':complete}}
 def run_phase(self,states,budget=15,pause_limit=60):
  clock=Clock();calls=[]
  def worker(path,body=None):
   calls.append((path,body))
   if body is not None:return {}
   return states(clock.now)
  with patch.object(task.time,'monotonic',clock.monotonic),patch.object(task.time,'sleep',clock.sleep),patch.object(task,'worker',worker),patch.object(task,'MAX_ROTATION_PAUSE',pause_limit):
   result=task.phase(JOB,UID,'training',budget)
  return result,calls,clock.now
 def test_rotation_does_not_spend_training_budget(self):
  result,calls,elapsed=self.run_phase(lambda t:self.state(paused=t<20,complete=t>=30))
  self.assertTrue(result['verified_complete']);self.assertEqual(elapsed,30)
  self.assertEqual(len([x for x in calls if x[1] is not None]),3)
 def test_ordinary_wait_retains_active_timeout(self):
  with self.assertRaises(TimeoutError):self.run_phase(lambda t:self.state())
 def test_permanent_rotation_still_has_wall_limit(self):
  with self.assertRaises(TimeoutError):self.run_phase(lambda t:self.state(paused=True),pause_limit=20)
 def test_untrusted_truthy_pause_cannot_extend_budget(self):
  def state(t):
   value=self.state();value['rotation_paused']='true';return value
  with self.assertRaises(TimeoutError):self.run_phase(state)
 def test_failure_remains_failure_during_rotation(self):
  def state(t):
   value=self.state(paused=True);value['phase']='failed';return value
  with self.assertRaises(ValueError):self.run_phase(state)
 def test_pipeline_has_bounded_overnight_envelope(self):
  spec=bridge.manifest(JOB)['spec'];self.assertEqual(spec['timeouts']['pipeline'],'18h0m0s');self.assertEqual(spec['timeouts']['tasks'],'17h50m0s')

if __name__=='__main__':unittest.main()
