from setuptools import setup

req_packages = """
gym~=0.19.0
numpy~=1.21.2
PyYAML~=5.4.1
scipy~=1.7.1
parse~=1.19.0
tensorboardX~=2.4
crc32c~=2.2
comet-ml~=3.15.3
"""

setup(
    name="fleetai",
    version="0.1",
    install_requires=req_packages.strip().split("\n"),
    packages=["fleetai", "fleetai.vec_env", "fleetai.ppo", "fleetai.actor_critic"],
    include_package_data=True,
    description="Using Reinforcement Learning to play the game Battleship"
)
