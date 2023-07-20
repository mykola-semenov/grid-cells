import torch
from torch.nn import Module


class GridTissue(Module):
    def __init__(self, n: int, 
        input_size: tuple[int, int], 
        step :int = 1, 
        V=None, 
        f_base=None, 
        preferred_direction=None,
        scale=None,
        phase_shift=None, 
    ):

        self.f_base = f_base
        if self.f_base is None:
            self.f_base = torch.rand((n,), requires_grad=True)

        self.V = V
        if self.V is None:
            self.V = torch.rand((1,2), requires_grad=True)

        self.preferred_direction = preferred_direction
        if self.preferred_direction is None:
            self.preferred_direction = self.rand((n,2), reqeuires_grad=True)
            self.preferred_direction = torch.nn.functional.normalize(self.preferred_direction)

        self.scale = scale
        if self.scale is None:
            self.scale = torch.rand((n,), requires_grad=True)

        self.phase_shift = phase_shift
        if self.phase_shift is None:
            self.phase_offset = torch.rand((n,), requires_grad=True)

        self.input_size = input_size
        scale_i = (input_size[0] - 1)/2
        scale_j = (input_size[1] - 1)/2
        range_i = torch.arange(-scale_i, scale_i + step, step)
        range_j = torch.arange(-scale_j, scale_j + step, step)
        self.X, self.Y = torch.meshgrid(range_i, range_j, "ij")
        self.XY = torch.stack([self.X, self.Y]).T

        self.n = n

    def forward(self, inp: torch.Tensor):
        # STUPPPPIIIIIID
        f_VCO = self.f_base + (
            self.scale * torch.dot(self.V.view(self.n,2), self.preferred_direction)
            )

        neurons_directions = torch.sum(self.XY * self.preferred_direction, axis=1)

        term_1 = torch.cos(2*torch.pi*self.f_basebase * neurons_directions)
        term_2 = torch.cos(2*torch.pi*f_VCO * neurons_directions)

        return term1 + term2 + self.phase_shift
        