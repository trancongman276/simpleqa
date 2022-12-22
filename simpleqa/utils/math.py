# -----------------------------------------------------------------------------
# MIT License
#
# Copyright (c) 2022 Gradients Technologies. All rights reserved.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----------------------------------------------------------------------------

import torch

def norm(x: torch.Tensor, dim: int = 1, **kwargs) -> torch.Tensor:
    """
    Normalize a tensor.
    :param x:       input torch tensor.
    :param dim:     dimension to be normalized.
    :param kwargs:  additional keyword arguments.
    :return:        normalized torch tensor.
    """
    # Make sure input is torch tensor.
    x = x if isinstance(x, torch.Tensor) else torch.Tensor(x)
    # Return result.
    return x / x.norm(dim=dim)[:, None]

def cosine(a: torch.Tensor, b: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Calculate cosine between two tensor.
    :param a:       first input as a torch tensor.
    :param b:       second input as a torch tensor.
    :param kwargs:  additional keyword arguments.
    :return:        cosine matrix as a torch tensor.
    """
    # Make sure inputs is torch tensors.
    a = a if isinstance(a, torch.Tensor) else torch.Tensor(a)
    b = b if isinstance(b, torch.Tensor) else torch.Tensor(b)
    # Return result.
    return torch.mm(a, b.transpose(0, 1))