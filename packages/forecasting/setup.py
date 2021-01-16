from distutils.core import setup

setup(
	name = 'pmforecast',
	version = '0.0.1',
	packages = ['pmforecast'],
	long_description = open('README.md').read(),
	install_requires = [
		'pandas',
		'numpy>=1.18.1',
		'simplejson',
		'configparser',
		'psycopg2-binary'


	]
)
