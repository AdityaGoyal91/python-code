from distutils.core import setup

setup(
	name = 'pmutils',
	version = '0.4',
	packages = ['pmutils'],
	long_description = open('README.md').read(),
	install_requires = [
		'pandas',
		'numpy',
		'simplejson',
		'configparser',
		'psycopg2-binary',
		'sshtunnel',
		'keyring',
		'sqlalchemy'
	]
)
