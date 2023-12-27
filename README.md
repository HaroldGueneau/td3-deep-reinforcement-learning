# TD3 - Deep Reinforcement Learning

After the DDPG, some improvements have been proposed, by considering them together this became a new model, called
Twin-Delayed DDPG (TD3) [1].
Please find theoretical information about the TD3 model in the td3.pdf file.

## Scripts

* launcher_grid_calibration.py : Grid search for hyperparameters calibration.
* launcher_agent_training.py : Complete training of an agent.
* launcher_optimal_agent_playing.py : To see an optimal agent interacting with the environment.

## Running the scripts with Docker

1. Install [Docker Engine](https://docs.docker.com/engine/install/).
2. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) (you may skip this step).
3. Make sure `Make` is installed and available on CLI.
4. Run any of the commands below.

* launcher_grid_calibration.py : `make grid`.
* launcher_agent_training.py : `make train`.
* launcher_optimal_agent_playing.py : `make play`.

A container with respective name will be created in Docker, wait for its execution to finish and check the results in
container logs.

> If for any reason you cannot run `Make` command, simply copy/paste content of the command needed from the `Makefile`.

## Running without Docker

1. Install Python 3.8.
2. Install pip if not installed already and make sure it's available on the CLI.
3. Run `pip install -r requirements.txt` in the project folder to install necessary packages.
4. Run `python3.8 file_name` or `py3.8 file_name` if on Windows, replacing `file_name` with any file mentioned earlier.

## Environment

The chosen environment is the [Gym 'Pendulum-v0'](https://github.com/openai/gym/wiki/Pendulum-v0), a simple one where
the agent has to swing up an inverted pendulum.
In order to transpose the code into other environments, a modular implementation has been proposed. The main point to
manage in order to do it is the Preprocessor class, and to adjust the hyperparameters.

## Hyperparameters Calibration

A simple hyperparameter optimization has been performed, a grid search on the learning rate for the three neural
networks. With the current code, it is easy to perform an optimization on other hyperparameters if necessary.

## Data Structure

```text
                    The environment will provide
+-------------+     a new state and a new reward       +---------------------------------------------+
| Environment |     at each new step                   |                TD3 Agent                    |
+-------------+                                        +---------------------------------------------+
|             +----------------------------------------> +-----------------------------------------+ |
|             |                                        | |      Actor       |        Critic        | |
|             <----------------------------------------+ +------------------+----------------------+ |
+-------------+     The actor will choose an action    |                                             |
                    with respect to the state          |                                             |
                                                       +------------------+--^-----------------------+
                                                                          |  |
               +                                  +   The agent will store|  |
               |                                  |   new experiences in  |  | The agent will ask for
               +----------------------------------+   the memory when     |  | past experiences in
                    If necessary, the                 tackling a step     |  | order to learn
                    communication with the                                |  |
                    environment will require          (PyTorch tensors)   |  |
                    to convert in PyTorch tensors                         |  | (PyTorch tensors)
                    the informations at each step,                        |  |
                    and vice versa                    +-------------------v--+----------------------+
                                                      |                  Memory                     |
                                                      +---------------------------------------------+
                                                      |                                             |
                                                      |                                             |
                                                      +---------------------------------------------+

                                                          The memory only interact with the agent
                                                          the experience will be stored as PyTorch
                                                          tensors
                        
```

## Results

The current optimal agent has a mean score over 100 episode oscillating in general between -150 and -170.

## Future Work

It seems that some of the hyperparameters have a huge influence on the convergence speed.
It would be suitable in the future to explore further hyperparameters optimization methods, such bayesian optimisation.
However, this is very challenging due to the high dimension of the hyperparameters space (Agent's hyperparameters + 3
Neural Networks ones).
A rigorous hyperparameters optimization would require a lot of computing power.

Another possible path is to implement a Prioritized Experience Replay in order to accelerate the convergence by choosing
the right sample during the training process.

## References

[1] [Fujimoto, Scott and Hoof, Herke and Meger, David. Addressing Function Approximation Error in Actor-Critic Methods. In International Conference on Machine Learning, 2018.](https://arxiv.org/pdf/1802.09477.pdf)
