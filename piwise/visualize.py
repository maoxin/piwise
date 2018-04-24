import numpy as np

from torch.autograd import Variable

from visdom import Visdom

class Dashboard:

    def __init__(self, port):
        self.vis = Visdom(port=port)

    def loss(self, losses, title='loss_over_epoch'):
        x = np.arange(1, len(losses)+1, 1)
        losses = np.array(losses)

        self.vis.line(losses, x, env='loss_all_epoch', opts=dict(title=title), win=title)

    def image(self, image, title, env):
        if image.is_cuda:
            image = image.cpu()
        if isinstance(image, Variable):
            image = image.data
        image = image.numpy()

        self.vis.image(image, env=f'images_{env}', opts=dict(title=title), win=title)

    def score(self, scores, title='IoU_over_epoch'):
        x = np.arange(1, len(scores)+1, 1)
        scores = np.array(scores)

        self.vis.line(scores, x, env='scores_all_epoch', opts=dict(title=title), win=title)