import torch
from torch.nn import Module


class GridTissue(Module):
  def __init__(self, 
    n, 
    n_dir=3,
    threshold=0.5,
    dir_vec=None,
    phase=None,
    scale=None,
  ):
    """
    Args:
      n (int): number of neurons
    """

    super().__init__()
    self.n = n
    self.n_dir = n_dir

    # Orientation angle
    self.dir_vec = dir_vec
    if self.dir_vec is None:
      self.dir = torch.rand((n, self.n_dir), requires_grad=True)
      self.dir_vec = torch.rand((n, self.n_dir, 2))
    self._ori2vec()

    # phase shift
    self.phase = phase
    if self.phase is None:
      self.phase = torch.rand((n,3), requires_grad=True)

    # scale
    self.scale = scale
    if self.scale is None:
      self.scale = torch.rand((n,1), requires_grad=True).expand((n, n_dir))

    # threshold
    self.threshold = threshold
    

  def _ori2vec(self):
    self.dir_vec[:,:,0] = torch.cos(self.dir)
    self.dir_vec[:,:,1] = torch.sin(self.dir)

  def forward(self, pos):
    self._ori2vec()
    res = torch.cos(self.scale * torch.tensordot(pos, self.dir_vec, dims=[[1],[2]]) + self.phase)
    res = torch.prod(res, axis=-1)
    res = torch.sigmoid(res + self.threshold)
    return res
