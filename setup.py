from setuptools import setup, find_packages

setup(name='stribor',
      version='0.1.0',
      description='normalizing flows and neural flows',
      packages=find_packages('.'),
      packages=setuptools.find_packages(),
      install_requires=['numpy>=1.20.1', 'torch>=1.8.0', 'torchdiffeq==0.2.1']
      python_requires='>=3.6',
      zip_safe=False
)
