import json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import providers

class TrafficTests(unittest.TestCase):
 def test_background_primary_uses_separate_key_and_falls_back_on_night_rejection(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)
   for name in ('primary-key','backup-key','router-background-key'):(root/name).write_text(name+'-fixture')
   (root/'fallback.json').write_text(json.dumps({'base_url':'https://fallback.example','model':'fixture','token':'fixture'}))
   calls=[]
   def http(url,body=None,headers=None,timeout=None):
    calls.append((url,headers))
    if 'router' in url:
     self.assertEqual(headers['Authorization'],'Bearer '+'router-background-key-fixture')
     if url.endswith('/health'):return {}
     raise providers.Unavailable('night_background_paused')
    if 'backup' in url:raise providers.Unavailable('backup_unavailable')
    return {'content':[{'type':'text','text':'{"action":"observe"}'}]}
   with patch.dict(os.environ,{'BACKGROUND_ROUTER_URL':'http://router','BACKGROUND_ROUTER_KEY_FILE':str(root/'router-background-key'),'BACKUP_URL':'http://backup'}),patch.object(providers,'http',side_effect=http):
    result,info=providers.Models({},d).plan('fixture',{})
   self.assertEqual(result,{'action':'observe'});self.assertEqual(info['provider'],'fallback')
   self.assertFalse(any('glm-primary' in u for u,_ in calls))

if __name__=='__main__':unittest.main()
