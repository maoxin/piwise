import numpy as np

from torch.autograd import Variable

from visdom import Visdom

class Dashboard:

    def __init__(self, port):
        self.vis = Visdom(port=port)

    def loss(self, losses, title, env):
        x = np.arange(1, len(losses)+1, 1)
        losses = np.array(losses)

        # self.vis.line(np.array(losses), x, env='loss', opts=dict(title=title), win=title)
        self.vis.line(losses, x, env=f'loss_{env}', opts=dict(title=title), win=title)

    def image(self, image, title, env):
        if image.is_cuda:
            image = image.cpu()
        if isinstance(image, Variable):
            image = image.data
        image = image.numpy()

        # self.vis.image(image, env='images', opts=dict(title=title))
        self.vis.image(image, env=f'images_{env}', opts=dict(title=title))