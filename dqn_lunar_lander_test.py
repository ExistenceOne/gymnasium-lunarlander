import gymnasium as gym
import os
from qnet_lunar_lander import QNet

from dqn_train_test import DqnTester

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

def main():
    ENV_NAME = "LunarLander-v3"

    test_env = gym.make(ENV_NAME, render_mode="rgb_array")

    qnet = QNet(n_features=8, n_actions=4)

    dqn_tester = DqnTester(
        env=test_env, qnet = qnet, env_name=ENV_NAME, current_dir=CURRENT_DIR
    )
    dqn_tester.test()

    test_env.close()

if __name__ == "__main__":
    main()