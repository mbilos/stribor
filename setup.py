from setuptools import setup, find_packages

install_requires = [
    'numpy',
    'pytest',
    'scipy',
    'torch>=1.8.0',
    'torchdiffeq==0.2.1'
]

with open('README.md', 'r') as f:
    long_description = f.read()

setup(name='stribor',
      version='0.1.0',
      description='Library for normalizing flows and neural flows',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/mbilos/stribor',
      author='Marin Bilos',
      author_email='bilos@in.tum.de',
      packages=find_packages(),
      install_requires=install_requires,
      python_requires='>=3.6',
      zip_safe=False,
)
