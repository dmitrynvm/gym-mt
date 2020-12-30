class Team(object):
    metadata = {
        'strategy': ('fixed', 'random', 'pursuit', 'evasion')
    }

    def __init__(self, size, label, strategy):
        self.size = size
        self.label = label
        self.strategy = strategy

    def __repr__(self):
        return '{}({})'.format(self.label.capitalize(), str(self.size))

    def __str__(self):
        return '{}({})'.format(self.label.capitalize(), str(self.size))


class Pred(Team):
    def __init__(self, size):
        super().__init__(size, 'pred', 'pursuit')


class Prey(Team):
    def __init__(self, size):
        super().__init__(size, 'prey', 'fixed')


class Grid(object):
    def __init__(self, *shape, label='Grid'):
        self.shape = shape
        self.label = label
        self.actions = ('stop', 'up', 'down', 'left', 'right')

    def __repr__(self):
        return '{}{}'.format(self.label, self.shape)
