from setuptools import find_packages, setup

setup(

    name='vevestaX',
    packages=find_packages(include=['vevestaX']),
    version='5.5.0',
    description='Stupidly simple library to track machine learning experiments as well as features',
    author='Vevesta Labs',
    license='Apache 2.0',
    install_requires=['pandas','Jinja2','ipynbname','datetime','openpyxl','xlrd','requests','matplotlib','pyspark','numpy','scipy','statistics'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',

)
