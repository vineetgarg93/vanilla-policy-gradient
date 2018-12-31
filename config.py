class Config(object):
    """docstring for Config"""
    def __init__(self):
        # super(Config, self).__init__()
        # self.arg = arg

        self.num_epsiodes = 500
        self.max_length = 200
        self.learning_rate = 1e-3
        self.discount = 0.99