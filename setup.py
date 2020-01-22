from setuptools import setup

dist = setup(name="PanoramAI",
             author="Tom McClintock",
             author_email="thmsmcclintock@gmail.com",
             version="0.1.0",
             description="Using GANs to create panoramic images.",
             url="https://github.com/tmcclintock/PanoramAI",
             packages=['PanoramAI'],
             install_requires=['numpy', 'scipy']
)
