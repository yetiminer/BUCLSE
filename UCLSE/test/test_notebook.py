import os
import subprocess
import tempfile
import pathlib
import nbformat
from UCLSE.test.utils import yamlLoad
from nbconvert.preprocessors import ExecutePreprocessor
import warnings

cwd=os.getcwd()

fixture_name=os.path.join(cwd,'UCLSE','test','fixtures','nb_fix.yml')
fixtures=yamlLoad(fixture_name)

# def _notebook_run(path):
	# """Execute a notebook via nbconvert and collect output.
	   # :returns (parsed nb object, execution errors)
	# """
	# assert os.path.isfile(path)
	# #dirname, __ = os.path.split(path)
	# #os.chdir(dirname)
	# tempfile.tempdir=os.getcwd()
	# with tempfile.NamedTemporaryFile(suffix=".ipynb",delete=False) as fout:
		# args = ["jupyter", "nbconvert", "--to", "notebook", "--execute",
		  # "--ExecutePreprocessor.timeout=60",
		  # "--output", fout.name, path]
		# print(args)
		# print(pathlib.Path(fout.name).exists())
		# print(pathlib.Path(path).exists())
		

		# subprocess.check_call(args) #breaks here when there are errors in the notebook

		# fout.seek(0)
		# name=fout.name
		
		# #print(fout.read())
		# nb = nbformat.read(name, nbformat.current_nbformat,allow_errors=True)
	
	# os.remove(name) #this is necessary with windows, otherwise delete=False is not needed
	# errors = [output for cell in nb.cells if "outputs" in cell
					 # for output in cell["outputs"]\
					 # if output.output_type == "error"]
		

	# return nb, errors
	
def _notebook_run(notebook_path):
	nb_name, _ = os.path.splitext(os.path.basename(notebook_path))
	dirname = os.path.dirname(notebook_path)

	with open(notebook_path) as f:
		nb = nbformat.read(f, as_version=4)

	proc = ExecutePreprocessor(timeout=600, kernel_name='python3',allowerrors=True)
	proc.allow_errors = True

	proc.preprocess(nb, {'metadata': {'path': dirname}}) #critically the notebooks are setup to run from their local directory so this needs to be specified
	output_path = os.path.join(dirname, '{}_all_output.ipynb'.format(nb_name))

	with open(output_path, mode='wt') as f:
		nbformat.write(nb, f)

	errors = {}
	for cell in nb.cells:
		
		if 'outputs' in cell:
			for output in cell['outputs']:
				if output.output_type == 'error':
					errors[cell['execution_count']]=output
					
	os.remove(output_path)

	return nb, errors


def _test_ipynb(path):
    nb, errors = _notebook_run(path)
    assert errors == {}
	
def process_through_all_nbs_in_directory(pa_dir):

	notebooks=list(filter(lambda x: x.endswith(".ipynb"),os.listdir(pa_dir)))
	os.chgdir(pa_dir)

	for nb in notebooks:
		print(nb)
		_test_ipynb(nb)

def test_notebooks():
	

	for fixture in fixtures:
		path=fixture['path']
		fix_errors=fixture['errors']
		_,errors=_notebook_run(path)
		for k,val in errors.items():
			try:
				assert val['ename']==fix_errors[k]['ename']
				assert val['evalue']==fix_errors[k]['evalue']
				
				
			except KeyError:
				print('Error not in fixture file, cell number', str(k))
				
			except AssertionError:
				print('Error not matching fixture value, cell number', str(k), ' expecting ',val['evalue'] )
		print(path, ' Done')
				
def test_fixture_presence():
	pa_dir='Tutorials'
	notebooks=list(filter(lambda x: x.endswith(".ipynb"),os.listdir(pa_dir)))
	missing=[]

	fixture_paths=[os.path.basename(fixture['path']) for fixture in fixtures]


	for nb in notebooks:
		try:
			assert nb in fixture_paths
		except AssertionError:
			missing.append(nb)

	if len(missing)>0:
		warnings.warn(UserWarning('missing notebook fixtures ', missing))
		
			
			