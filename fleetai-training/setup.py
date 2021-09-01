from setuptools import setup

req_packages = ["gym==0.19.0", "numpy==1.21.2", "PyYAML==5.4.1", "scipy==1.7.1",
                "tensorboard~=2.6.0", "parse~=1.19.0"]

setup(
    name="fleetai",
    version="0.1",
    install_requires=req_packages,
    packages=["fleetai", "fleetai.vec_env", "fleetai.ppo", "fleetai.actor_critic"],
    include_package_data=True,
    description="Using Reinforcement Learning to play the game Battleship"
)
