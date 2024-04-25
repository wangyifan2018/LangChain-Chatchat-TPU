# -*- coding: utf-8 -*-
#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
from langchain.embeddings.base import Embeddings
from typing import List
from .sentence_model import SentenceModel


class Word2VecEmbedding(Embeddings):
    def __init__(self, model_name, dev_id):
        self.model = SentenceModel(model_name_or_path=model_name, dev_id=dev_id)

    def embed_query(self, text: str) -> List[float]:
        embeddings_tpu = self.model.encode_tpu(text)
        return embeddings_tpu.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings_tpu = self.model.encode_tpu(texts)
        return embeddings_tpu.tolist()

