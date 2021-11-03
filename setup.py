from setuptools import find_packages, setup
setup(
    name='vevestaX',
    packages=find_packages(include=['vevestaX']),
    version='0.1.0',
    description='Track failed and successful experiments as well as features',
    author='Me',
    license='MIT',
    install_requires=['pandas','ipynbname','datetime','openpyxl','xlrd'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)
