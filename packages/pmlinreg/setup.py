from distutils.core import setup

setup(
	name = 'pmlinreg',
	version = '0.0.1',
	packages = ['pmlinreg'],
	long_description = open('README.md').read(),
	install_requires = [
		'pandas',
        'numpy',
        'psycopg2',
        'os',
        'datetime',
        'pyspark.sql.types',
        'sklearn',
        'math',
        'matplotlib',
        'seaborn',
        'simplejson',
        'configparser'

	]
)
