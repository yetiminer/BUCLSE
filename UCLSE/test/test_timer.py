from UCLSE.custom_timer import CustomTimer
import pytest
import os
from UCLSE.test.utils import yamlLoad
error_dic={'AssertionError':AssertionError,'TypeError':TypeError}

def test_inputs():
		pa=os.getcwd()
		config_name='UCLSE\\test\\fixtures\\timer_cfg.yml'
		config_path=os.path.join(pa,config_name)

		fixture_list=yamlLoad(config_path)
	
		for fixture in fixture_list:
			error=fixture.pop('error')
			with pytest.raises(error_dic[error]):
				
				timer=CustomTimer(**fixture)
			