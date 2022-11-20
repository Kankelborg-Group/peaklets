from setuptools import setup, find_packages

setup(
    name='peaklets',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pytest',
        'numpy',
        'numba',
    ],
    url='https://github.com/Kankelborg-Group/peaklets',
    license='',
    author='Charles C. Kankelborg',
    author_email='kankel@montana.edu',
    description='Decompose a 1D positive signal using only positive basis functions.'
)
