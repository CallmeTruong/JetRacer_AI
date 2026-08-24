from setuptools import setup, find_packages

setup(
    name='jetracer_ai',
    version='1.0.0',
    description='JetRacer AI Unified Autonomous Driving Platform (Lane Following & Urban Traffic Signals)',
    author='JetRacer AI Team',
    packages=find_packages(include=['jetracer_ai', 'jetracer_ai.*', 'config', 'config.*']),
    install_requires=[
        'numpy',
        'opencv-python',
        'requests'
    ],
    python_requires='>=3.6',
)
