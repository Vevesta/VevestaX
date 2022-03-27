from setuptools import find_packages, setup
setup(
    name='vevestaX',
    packages=find_packages(include=['vevestaX']),
    version='2.5.0',
    description='Track failed and successful machine learning experiments as well as features',
    author='Vevesta Labs',
    license='Apache 2.0',
    install_requires=['pandas','ipynbname','datetime','openpyxl','xlrd'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)
