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

import os
import torch
import pickle
import hashlib
import pathlib
import pandas as pd
from typing import Union
from simpleqa.utils import math
from simpleqa.responders import Responder


class LocalResponder(Responder):
    """
    This responder will give answers from given local datasets (xlsx and csv format).
    """

    def __init__(self, data = "data", **kwargs) -> None:
        """
        Class constructor.
        :param data:    path to data file or directory.
        :param kwargs:  additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.emb, self.ans = self.prepare(data=data)

    def prepare(self, data = "data", prepare: str = "prepare", **kwargs):
        """
        Prepare datasets and embeddings. 
        :param data:    path to data file or directory.
        :param prepare: path to prepare directory.
        :param kwargs:  additional keyword arguments.
        :return:        question embeddings and answer data.
        """
        hash = hashlib.md5(self.model_name.encode()).hexdigest()
        # Path to prepare folder and files.
        pre_dir = pathlib.Path(prepare)
        pre_dir.mkdir(parents=True, exist_ok=True)
        emb_file = pre_dir.joinpath("{}.emb".format(hash))
        ans_file = pre_dir.joinpath("{}.ans".format(hash))
        # Load and return data if pre-generated files exist.
        if ans_file.is_file() and emb_file.is_file():
            return torch.load(str(emb_file)), pickle.load(ans_file.open("rb"))
        # Load dataset.
        data = self.load_data(data)
        qs, ans = data["QUESTION"].tolist(), data["ANSWER"].tolist()
        qs_emb = self.encode(qs)
        # Save pre-generated files.
        torch.save(qs_emb, str(emb_file))
        pickle.dump(ans, ans_file.open("w+b"))
        # Return result.
        return qs_emb, ans

    @staticmethod
    def load_data(data: Union[list[str], str], **kwargs) -> pd.DataFrame:
        """
        Load data from files or directory.
        :param data:    path to data files or directory.
        :param kwargs:  additional keyword arguments.
        :return:        loaded data as pandas data-frame.
        """
        # Validate path types.
        assert isinstance(data, str) or isinstance(data, list), "Invalid data path!"
        # In case path is singular string.
        if isinstance(data, str):
            path = pathlib.Path(data)
            # Validate path, only file or directory is accepted.
            assert path.is_file() or path.is_dir(), "Invalid data path!"
            # If path is file.
            if path.is_file():
                # Validate file path
                assert path.suffix in [".xlsx", ".csv"], "Invalid data file!"
                # Load data file
                df = pd.read_excel(str(path)) if path.suffix == ".xlsx" else pd.read_csv(str(path))
            # In case path is directory, recursive load all sub-files.
            elif path.is_dir():
                df = LocalResponder.load_data([str(i) for i in path.glob("**/*") if i.suffix in [".xlsx", ".csv"]])
        # In case path is list of files or directories, recursive load them all and then concat them.
        elif isinstance(data, list):
            df = [LocalResponder.load_data(file) for file in data]
            df = pd.concat(df)
        # Preprocess data-frame.
        df = df.apply(lambda x: x.str.split("\n"))
        df = df.explode(["QUESTION"]).explode(["ANSWER"])
        # Return result.
        return df

    def answer(self, inputs: Union[list[str], str], **kwargs) -> list:
        """
        Give answer based on input questions.
        :param inputs:  input questions.
        :param kwargs:  additional keyword arguments.
        :return:        answers.
        """
        embs = self.encode(inputs)
        cosine = math.cosine(embs, self.embs)
        probs, idxs = torch.max(cosine, dim=1)
        responses = [[self.ans[idxs[i]], probs[i]] for i in range(len(inputs))]
        return responses
        