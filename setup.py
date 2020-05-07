from setuptools import setup

setup(
    name='RLC',
    version='0.4',
    packages=['RLC', 'RLC.move_chess', 'RLC.capture_chess', 'RLC.capture_chess_rllib', 'RLC.real_chess'],
    url='https://github.com/irvenae/RLC',
    license='MIT',
    author='a.groen, irvenae',
    author_email='arjanmartengroen@gmail.com',
    description='Collection of reinforcement learning algorithms, all applied to chess or chess related problems'
)
