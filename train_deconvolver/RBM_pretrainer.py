import torch

from matplotlib import pyplot
from torchvision import transforms




def show_images(images, cols = 1, titles = None):
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = pyplot.figure()
    rows = n_images // cols
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, rows, n + 1)
        if image.ndim == 2:
            pyplot.gray()
        pyplot.imshow(image)
        pyplot.axis('off')
        # a.set_title(title)
    # fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    pyplot.show()

'''
def show_reconstruct(model):
    xs = []
    rs = []
    for i in range(8):
        x = (test_set[i][0][0] > 0).float()
        with torch.no_grad():
            rx = torch.sigmoid(model.reverse(torch.sigmoid(model.forward(x.view(1, -1)))))
        xs.append(x.numpy())
        rs.append(rx.view(28, 28).numpy())

    print("input")
    show_images(xs)
    print("reconstruct")
    show_images(rs)
'''



class BernoulliRBM(torch.nn.Module):
    """Bernouli Restricted Boltzmann Machine
    http://deeplearning.net/tutorial/rbm.html
    """

    def __init__(self, n_visible, n_hidden  = 500):
        super(BernoulliRBM, self).__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.weight = torch.nn.Parameter(torch.Tensor(self.n_visible, self.n_hidden))
        self.vbias = torch.nn.Parameter(torch.Tensor(self.n_visible))
        self.hbias = torch.nn.Parameter(torch.Tensor(self.n_hidden))
        self.reset_parameters()

    def plot_weight(self, n=64, rows=8):
        print 'Not implemented'
        #show_images(model.weight.t()[:n].view(n, 28, 28).data.numpy(), rows)

    def forward(self, v):
        return torch.addmm(1, self.hbias, 1, v, self.weight)

    def reverse(self, h):
        return torch.addmm(1, self.vbias, 1, h, self.weight.t())

    def reset_parameters(self):
        a = 4.0 * ((6.0 / (self.n_hidden + self.n_visible)) ** 0.5)
        self.weight.data.uniform_(-a, a)
        self.hbias.data.zero_()
        self.vbias.data.zero_()

    def sample(self, input):
        return torch.bernoulli(torch.sigmoid(input))

    def free_energy(self, v):
        vterm = v.mv(self.vbias)
        hterm = torch.log(1 + self.forward(v).exp()).sum(1)
        return -hterm - vterm

    def contrastive_divergence(self, v, h_sample=None, n_iter=1):
        with torch.no_grad():
            if h_sample is None:
                h_sample = self.sample(self.forward(v))
            for k in range(n_iter):
                pre_v = self.reverse(h_sample)
                v_sample = self.sample(pre_v) # https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf 3.2 -> Dice che è indifferente rifare o meno il sample su v
                h_sample = self.sample(self.forward(v_sample)) #https://github.com/wiseodd/generative-models/blob/master/RBM/rbm_binary_cd.py In questa implementatione
                                                               #il sampling non viene fatto l'ultima volta
            ce = torch.nn.functional.binary_cross_entropy(torch.sigmoid(pre_v), v)
        loss = torch.mean(self.free_energy(v)) - torch.mean(self.free_energy(v_sample))
        return loss, h_sample, ce

    def pretrain(self, X):
        #configurations
        n_batch = 32
        n_epoch = 10
        n_hidden = 500
        percistent = False
        n_gibbs = 3
        lr = 0.01
        #todo modificare
        n_input = len(X)
        model = BernoulliRBM(n_input, n_hidden)
        optim = torch.optim.SGD(model.parameters(), lr)
        #todo il train loader va implementato
        train_loader=[]
        for epoch in range (n_epoch):
            sum_loss = 0
            hs = None
            for i, (x, t) in enumerate(train_loader, 1):
                x = (x > 0).float()
                model.zero_grad()
                loss, hs, ce = model.contrastive_divergence(x.view(-1, n_input), hs, n_gibbs)
                if not percistent:
                    hs = None
                loss.backward()
                optim.step()
                sum_loss += ce.item()
            print("epoch: {}, loss: {}".format(epoch, sum_loss / len(train_loader)))
            model.plot_weight()


class GaussianRBM(torch.nn.Module):
    """Gaussian-Bernoulli Restricted Boltzmann Machine
    https://users.ics.aalto.fi/praiko/papers/icann11.pdf
    https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2012-95.pdf
    """

    def __init__(self, n_visible, n_hidden):
        super(GaussianRBM, self).__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.weight = torch.nn.Parameter(torch.Tensor(self.n_visible, self.n_hidden))
        self.vbias = torch.nn.Parameter(torch.Tensor(self.n_visible))
        self.hbias = torch.nn.Parameter(torch.Tensor(self.n_hidden))
        self.logvar = torch.nn.Parameter(torch.Tensor(self.n_visible))
        self.reset_parameters()

    def plot_weight(self, n=64, rows=8):
        print('Not Implemented')
        #show_images(model.weight.t()[:n].view(n, 28, 28).data.numpy(), rows)

    def forward(self, v):
        return torch.addmm(1, self.hbias, 1, v / self.var, self.weight)

    def reverse(self, h):
        return torch.addmm(1, self.vbias, 1, h, self.weight.t())

    @property
    def var(self):
        return self.logvar.exp()

    def reset_parameters(self):
        a = 4.0 * ((6.0 / (self.n_hidden + self.n_visible)) ** 0.5)
        self.weight.data.uniform_(-a, a)
        self.hbias.data.zero_()
        self.vbias.data.zero_()
        self.logvar.data.fill_(1)

    def sample_hidden(self, v):
        return torch.bernoulli(torch.sigmoid(self.forward(v)))

    def sample_visible(self, h):
        m = self.reverse(h)
        return torch.normal(m, self.var.expand_as(m))

    def free_energy(self, v):
        vb = v - self.vbias
        vterm = 0.5 * (vb * vb).sum(1)
        hterm = torch.log(1 + self.forward(v).exp()).sum(1)
        return -hterm - vterm

    def contrastive_divergence(self, v, h_sample=None, n_iter=1):
        with torch.no_grad():
            if h_sample is None:
                h_sample = self.sample_hidden(v)
            for k in range(n_iter):
                v_sample = self.sample_visible(h_sample)
                h_sample = self.sample_hidden(v_sample)
            err = torch.nn.functional.mse_loss(v_sample, v)
        loss = torch.mean(self.free_energy(v)) - torch.mean(self.free_energy(v_sample))
        return loss, h_sample, err

    def pretrain(self, X):
        # config
        n_batch = 128
        n_epoch = 10
        n_hidden = 1000
        percistent = False
        n_gibbs = 1
        lr = 1e-6
        #todo modificare
        n_input = len(X)
        model = GaussianRBM(n_input, n_hidden)
        #todo trainloader da implementare
        train_loader =[]
        optim = torch.optim.SGD(model.parameters(), lr)
        print(len(train_loader))
        for e in range(n_epoch):
            sum_loss = 0
            hs = None
            for i, (x, t) in enumerate(train_loader, 1):
                model.zero_grad()
                loss, hs, ce = model.contrastive_divergence(1 - x.view(-1, n_input), hs, n_gibbs)
                if not percistent:
                    hs = None
                loss.backward()
                optim.step()
                sum_loss += ce.item()
            print("epoch: {}, loss: {}".format(e, sum_loss / len(train_loader)))
            model.plot_weight()
            #show_reconstruct(model)









