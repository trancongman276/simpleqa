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
from typing import Union
from simpleqa.utils import math
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer


class Responder(ABC):
    """
    The Responder has responsibility to give answers for received questions 
    based on text similarity search on a given dataset.
    """

    @property
    def model(self):
        """
        Get current model.
        """
        return self._model
    
    @property
    def model_name(self):
        """
        Get current model name.
        """
        return self._name

    def __init__(self, model: str = 'all-MiniLM-L6-v2', **kwargs) -> None:
        """
        Class constructor.
        :param model:       name of pre-trained model.
        :param threshold:   threshold.
        :param kwargs:      additional keyword arguments.
        """
        self._name = model
        self._model = SentenceTransformer(model)

    def encode(self, inputs: Union[list[str], str], is_norm: bool = True, **kwargs) -> torch.Tensor:
        """
        Encode inputs into embeddings.
        :param inputs:  inputs to be encoded.
        :param is_norm: normalize outputs or not.
        :param kwargs:  additional keyword arguments.
        :return:        encoded as a torch tensor.
        """
        inputs = inputs if isinstance(inputs, list) else [inputs]
        outputs = self._model.encode(inputs)
        return math.norm(outputs) if is_norm else outputs

    def cosine(self, a: Union[list[str], str], b: Union[list[str], str], is_norm = True, **kwargs) -> torch.Tensor:
        """
        Calculate cosine between two input text.
        :param a:       first input text.
        :param b:       second input text.
        :param is_norm: normalize outputs or not.
        :param kwargs:  additional keyword arguments.
        :return:        cosine matrix as torch tensor.
        """
        a, b = a if isinstance(a, list) else a, b if isinstance(b, list) else b
        return math.cosine(self.encode(a, is_norm), self.encode(b, is_norm))

    @abstractmethod
    def answer(self, inputs: Union[list[str], str], **kwargs) -> list[str]:
        """
        Give answer based on input questions.
        :param inputs:  input questions.
        :param kwargs:  additional keyword arguments.
        :return:        answers.
        """
        raise NotImplementedError()
    