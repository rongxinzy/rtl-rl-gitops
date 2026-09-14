import contextlib,io,json,os,pathlib,tempfile,types,unittest
from unittest.mock import patch
import native_swanlab as n
class SDK:
 __version__='0.10.0'
 def __init__(self):self.calls=[];self.active=None
 def login(self,api_key,save):self.login_save=save;print(api_key)
 def init(self,**kw):self.calls.append(kw);self.active=kw['id']
 def Settings(self,**kw):return kw
 def Text(self,value):return value
 def log(self,*a,**kw):pass
 def finish(self,**kw):self.finished=kw
class Tests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.out=pathlib.Path(self.tmp.name);self.key=self.out/'key';self.key.write_text('PRIVATE_TEST_CREDENTIAL');self.proxy=self.out/'proxy';self.proxy.write_text(__import__('urllib.parse',fromlist=['urlunsplit']).urlunsplit(('http','fixture-user'+':'+'fixture-pass'+'@proxy:80','','','')));self.meta={'project':'RTL-RL','workspace':'krli','job_id':'l20-123','device_label':'l20'};self.training={'backend':'llamafactory','dataset_id':'a'*64,'unapproved_field':'NO_CONFIG_LEAK'}
 def tearDown(self):self.tmp.cleanup()
 def start(self,sdk,step=0,meta=None):
  with patch.dict(os.environ,{},clear=True):return n.start(self.out,'b'*64,meta or self.meta,self.training,step,sdk,self.key,self.proxy)
 def test_resume_uses_same_cloud_identity(self):
  sdk=SDK();r=self.start(sdk);r.finish('paused',1);self.start(sdk,1);self.assertEqual(sdk.calls[0]['id'],sdk.calls[1]['id']);self.assertTrue(all(x['resume']=='allow' for x in sdk.calls))
 def test_credential_never_saved_or_logged(self):
  sdk=SDK();output=io.StringIO()
  with contextlib.redirect_stdout(output):self.start(sdk)
  public=json.dumps(sdk.calls)+output.getvalue()+(self.out/'swanlab-native.json').read_text();self.assertNotIn('PRIVATE_TEST',public);self.assertNotIn('NO_CONFIG_LEAK',public);self.assertNotIn('fixture-pass',public);self.assertFalse(sdk.login_save)
 def test_metadata_and_capture_controls(self):
  sdk=SDK();self.start(sdk);kw=sdk.calls[0];self.assertEqual(kw['job_type'],'sft');self.assertIn('llamafactory',kw['tags']);self.assertEqual(kw['settings']['terminal']['proxy_type'],'none');self.assertFalse(kw['settings']['probe']['runtime']);self.assertTrue(kw['settings']['probe']['hardware'])
 def test_changed_destination_refused(self):
  self.start(SDK());changed={**self.meta,'project':'Other'}
  with self.assertRaisesRegex(ValueError,'binding changed'):self.start(SDK(),1,changed)
 def test_sdk_error_is_sanitized(self):
  sdk=SDK()
  def fail(**kw):raise RuntimeError('PRIVATE_TEST_CREDENTIAL')
  sdk.login=fail
  with self.assertRaisesRegex(RuntimeError,'^native telemetry initialization failed$'):self.start(sdk)
 def test_acceptance_metadata_cannot_look_like_training(self):
  self.training['acceptance_only']=True;sdk=SDK();self.start(sdk);kw=sdk.calls[0];self.assertEqual(kw['job_type'],'backend-acceptance');self.assertTrue(kw['config']['acceptance_only']);self.assertIn('no model training',kw['description'])
 def test_actual_hardware_fields_only(self):
  fake=types.SimpleNamespace(__version__='2.13.0',version=types.SimpleNamespace(cuda='13.0'),cuda=types.SimpleNamespace(is_available=lambda:True,device_count=lambda:1,get_device_properties=lambda i:types.SimpleNamespace(name='NVIDIA L20',total_memory=48*1024**3)))
  with patch.dict('sys.modules',{'torch':fake}):
   hardware=n.device_observation();self.assertEqual(hardware['visible_gpu_count'],1);self.assertEqual(hardware['visible_gpus'][0]['name'],'NVIDIA L20');self.assertEqual(hardware['cuda_version'],'13.0');self.assertEqual(n.device_observation(True)['visible_gpu_count'],0)
 def test_real_manager_device_label(self):
  p=self.out/'settings.json';p.write_text(json.dumps({**self.meta,'device_label':'L20 GPU0'}));self.assertEqual(n.settings(p)['device_label'],'L20 GPU0')
 def test_config_rejects_credentials(self):
  p=self.out/'settings.json';p.write_text(json.dumps({**self.meta,'api_key':'private'}))
  with self.assertRaises(ValueError):n.settings(p)
 def test_lf_native_callback_preserved_config_collection_disabled(self):
  cb=types.SimpleNamespace(_log_config=True,on_log=lambda:'original-native-log')
  original=lambda args:cb;tuner=types.SimpleNamespace(get_swanlab_callback=original);saved=n.install_callback_guard(tuner);result=tuner.get_swanlab_callback(types.SimpleNamespace(swanlab_api_key=None));self.assertIs(saved,original);self.assertIs(result,cb);self.assertFalse(result._log_config);self.assertEqual(result.on_log(),'original-native-log')
 def test_key_in_lf_arguments_rejected(self):
  tuner=types.SimpleNamespace(get_swanlab_callback=lambda args:None);n.install_callback_guard(tuner)
  with self.assertRaises(ValueError):tuner.get_swanlab_callback(types.SimpleNamespace(swanlab_api_key='private'))
 def test_old_training_config_stays_opt_in(self):
  import train
  a=types.SimpleNamespace(model='/model',max_steps=20,max_length=1024)
  self.assertNotIn('use_swanlab',train.config(a,self.out))
if __name__=='__main__':unittest.main()
