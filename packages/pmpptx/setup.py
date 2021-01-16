from distutils.core import setup

setup(
	name = 'pmpptx',
	version = '0.2',
	packages = ['pmpptx'],
	long_description = open('README.md').read(),
	install_requires = [
		'pandas',
		'numpy',
		'python-pptx',
		'pmutils'
	]
)
