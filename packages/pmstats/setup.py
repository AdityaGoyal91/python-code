from distutils.core import setup

setup(
	name = 'pmstats',
	version = '0.9.1',
	packages = ['pmstats', ],
	long_description = open('README.md').read(),
	install_requires = [
		'pandas',
		'statsmodels',
		'simplejson',
		'configparser',
		'psycopg2-binary',
		'requests',
		'numpy',
		'matplotlib',
		'seaborn',
        'plotly >= 3.6',
	]
)
