from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
    readme = f.read()

modules = ['gensim', 'sentencepiece', 'keras', 'keras-bert']

setup(
    name='text_vectorian',
    version='0.1.10',
    description='For getting token embedded vectors for NLP.',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='Hideki INOUE',
    author_email='hideki@inoue-kobo.com',
    url='https://github.com/lhideki/text-vectorian',
    license='MIT',
    install_requires=modules,
    packages=find_packages(exclude=('tests', 'docs', '.vscode')),
    package_data={
        'text_vectorian':
            ['config.yml']
    }
)