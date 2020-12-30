from gym.envs.registration import register
from .entities import Pred, Prey, Grid


for game in [
    [[Pred(2), Prey(2)], Grid(2, 2)],
    [[Pred(4), Prey(3)], Grid(5, 5)]
]:
    teams, grid = game
    team_shape = [team.size for team in teams]
    name = 'MultiTeam-{}v{}-{}x{}-v0'.format(*team_shape, *grid.shape)
    register(
        id=name,
        entry_point='gym_mt.envs.multi_team:MultiTeam',
        kwargs={
            'teams': teams,
            'grid': grid
        }
    )
