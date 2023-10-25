import setuptools

with open("README.md", 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='SEBA',
    version='0.2',
    author='Konrad Danielewski',
    author_email='kdanielewski@gmail.com',
    description='Simple Ephys-Behavior Analysis',
    include_package_data=True,
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent'
    ],
    python_requires='>=3.10',
    install_requires=[
        'numpy',
        'seaborn',
        'matplotlib',
        'scikit-learn',
        'scipy',
        'pandas',
        'tables',
        'statsmodels'
    ], 
    url='https://github.com/KonradDanielewski/SEBA'
)