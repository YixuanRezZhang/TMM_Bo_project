from setuptools import setup, find_packages

setup(
    name='TMM_Bo_project',
    version='0.1.0',
    description='A modular framework for Bayesian optimization with various sampling and acquisition functions.',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/bayesian_optimization',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'optuna',
        'xgboost',
        'lightgbm',
        'torch',
        'botorch',
        'gpytorch',
        'pygmo',
        'scikit-opt',
        'ray',
        'matplotlib'',
    ],
    entry_points={
        'console_scripts': [
            'TMM_Bo_project=main:main'
        ]
    },
)
