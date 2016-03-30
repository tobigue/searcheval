from distutils.core import setup

setup(
    name='searcheval',
    version='0.2.0',
    author='Tobias Guenther',
    author_email='searcheval@tobigue.de',
    packages=['searcheval', 'searcheval.test'],
    scripts=[],
    url='https://github.com/tobigue/searcheval',
    license='LICENSE.txt',
    description='Library implementing metrics and tools for the evaluation of search results (rankings)',
    long_description=open('README.rst').read(),
    install_requires=[],
)
