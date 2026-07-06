from setuptools import find_packages, setup

package_name = 'vlahost'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    package_data={'vlahost': ['html/*.html', 'html/*.htm']},
    include_package_data=True,
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/vlahost_server.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='marvin',
    maintainer_email='marvin@todo.todo',
    description='HTTP bridge between the robot (state + quadcam image) and a remote VLA model server (e.g. pi0)',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vlahost_client = vlahost.client:main',
            'vlahost_server = vlahost.server:main',
        ],
    },
)
