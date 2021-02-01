from utils.grid_calibrator import GridCalibrator


possibilities = {
    'actor_learning_rate': [1e-2, 5e-3, 1e-3],
    'critic_1_learning_rate': [1e-2, 5e-3, 1e-3],
    'critic_2_learning_rate': [1e-2, 5e-3, 1e-3],
}

import gym
env = gym.make('Pendulum-v0')

grid_calibrator = GridCalibrator(env, nb_iterations=3, nb_episodes=100, nb_time_steps=200, moving_average_period=50, possibilities=possibilities)
grid = grid_calibrator.calibrate_agent()
print(grid)
grid.to_csv('TD3/output/calibration_result.csv')