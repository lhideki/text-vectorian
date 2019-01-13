from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='text_vectorian',
    version='0.1.0',
    description='For getting token embedded vectors for NLP.',
    long_description=readme,
    author='Hideki INOUE',
    author_email='hideki@inoue-kobo.com',
    url='https://github.com/lhideki/text-vectorian',
    license=license,
    packages=find_packages(exclude=('tests', 'docs', '.vscode')),
    package_data={
        'text_vectorian':
            ['config.yml',
            'models/wikija_sentencepiece.model',
            'models/wikija_sentencepieced_word2vec.model']
    }
)