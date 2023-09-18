#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [ ]

test_requirements = ['pytest>=3', ]

setup(
    author="Ahmad Ali",
    author_email='a.ali@alethea.ai',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="This code is an experiment to duplicate persona of twitted use that for making a chatbot acting like the twitter persona. ",
    install_requires=requirements,
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='twitter_persona',
    name='twitter_persona',
    packages=find_packages(include=['twitter_persona', 'twitter_persona.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/.../twitter_persona',
    version='0.1.0',
    zip_safe=False,
)
