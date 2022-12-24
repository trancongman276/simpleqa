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

    def __init__(self, data = os.getenv("DATA_PATH", "data"), **kwargs) -> None:
        """
        Class constructor.
        :param data:    path to data file or directory.
        :param kwargs:  additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.emb, self.ans = self.prepare(data=data)

    def prepare(self, data = os.getenv("DATA_PATH", "data"), prepare: str = os.getenv("PREPARE_PATH", "prepare"), **kwargs):
        """
        Prepare datasets and embeddings. 
        :param data:    path to data file or directory.
        :param prepare: path to prepare directory.
        :param kwargs:  additional keyword arguments.
        :return:        question embeddings and answer data.
        """
        hash = hashlib.md5(self.name.encode()).hexdigest()
        # Path to prepare folder and files.
        pre_dir = pathlib.Path(prepare)
        pre_dir.mkdir(parents=True, exist_ok=True)
        emb_file = pre_dir.joinpath("{}.emb".format(hash))
        ans_file = pre_dir.joinpath("{}.ans".format(hash))
        # Load and return data if pre-generated files exist.
        if ans_file.is_file() and emb_file.is_file():
            return torch.load(str(emb_file)), pickle.load(ans_file.open("rb"))
        # Load dataset.
        data = self.load(data)
        qs, ans = data["QUESTION"].tolist(), data["ANSWER"].tolist()
        qs_emb = self.encode(qs)
        # Save pre-generated files.
        torch.save(qs_emb, str(emb_file))
        pickle.dump(ans, ans_file.open("w+b"))
        # Return result.
        return qs_emb, ans

    def load(self, data: Union[list[str], str], **kwargs) -> pd.DataFrame:
        """
        Load data from files or directory.
        :param data:    path to data files or directory.
        :param kwargs:  additional keyword arguments.
        :return:        loaded data as pandas data-frame.
        """
        # In case path is singular string.
        if isinstance(data, str):
            path_obj = pathlib.Path(data)
            # If path is file.
            if path_obj.is_file():
                ext, path_str = path_obj.suffix, str(path_obj)
                # Load into data-frame based on file format.
                if ext == ".xlsx":
                    df = pd.read_excel(path_str)
                elif ext == ".csv":
                    df = pd.read_csv(path_str)
                # Raise error if file format is invalid.
                else:
                    raise TypeError("Invalid data format!")
                # Return data-frame.
                return df 
            # In case path is directory, recursive load all sub-files.
            elif path_obj.is_dir():
                return self.load([str(i) for i in path_obj.glob("**/*") if i.suffix in [".xlsx", ".csv"]])
            # Raise error if path is invalid.
            raise TypeError("Invalid data path!")
        # In case path is list of files or directories, recursive load them all and then concat them.
        elif isinstance(data, list):
            df = [self.load(file) for file in data]
            df = pd.concat(df)
            return df
        # Raise error if path is invalid.
        raise TypeError("Invalid data path!")

    def answer(self, inputs: Union[list[str], str], thresh: float = ..., **kwargs) -> list:
        """
        Give answer based on input questions.
        :param inputs:  input questions.
        :param kwargs:  additional keyword arguments.
        :return:        answers.
        """
        emb = self.encode(inputs)
        cosine = math.cosine(emb, self.emb)
        prob, ids = torch.max(cosine, dim=0)
        res = [[self.ans[ids[i]], prob[i]] for i in range(len(prob)) if prob[i] >= thresh]
        return res
    