from setuptools import setup, find_packages

setup(name='reafference',
    version='0.0.2',
    description='',
    url='',
    author='research.nameless',
    author_email='research.nameless@gmail.com',
    license='GNU3',
    classifiers=[
    "Programming Language :: Python :: 3",
    "Operating System :: OS Independent",
    ],
    install_requires=[
        'gym[atari]==0.24',
        'autorom[accept-rom-license]', 
        'numpy', 
        'tqdm',
        'more_itertools',
        'torch',
        'torchvision',
        'torchinfo',
        'webdataset',
        'jupyterlab',
        'ipywidgets',
        'ipykernel',
        'scikit-image',
        'ipycanvas',
        'matplotlib',
        'seaborn',
        'pygame'],
    include_package_data=True,
    packages=find_packages())

print(find_packages())
