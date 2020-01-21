from setuptools import setup

dist = setup(name="GANorama",
             author="Tom McClintock",
             author_email="thmsmcclintock@gmail.com",
             version="0.1.0",
             description="Using GANs to create panoramic images.",
             url="https://github.com/tmcclintock/GANorama",
             packages=['GANorama'],
             install_requires=['numpy', 'scipy']
)
