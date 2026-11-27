"""TRN-09's check that PQN learns: CartPole-v1, CleanRL's values, 500,000 steps, three seeds.
The bar is a mean evaluation return of 400, a guess from the ticket."""

from algoarcade.training.config import PQN_CARTPOLE

EXPERIMENT = "TRN-09-cartpole"
BASE = PQN_CARTPOLE
ARMS = {"pqn": {}}
SEEDS = [0, 1, 2]
