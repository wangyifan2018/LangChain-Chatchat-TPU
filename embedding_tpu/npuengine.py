# -*- coding: utf-8 -*-
#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import sophon.sail as sail
import numpy as np
from tqdm import tqdm
from configs import logger

class EngineOV:
    def __init__(self, model_path="./bmodel/text2vec-bge-large-chinese/bge_large_512_fp16_1b.bmodel", device_id=0) :
        self.net = sail.Engine(model_path, device_id, sail.IOMode.SYSIO)
        logger.info("load {} success, dev_id {}".format(model_path, device_id))
        self.graph_name = self.net.get_graph_names()[0]
        self.input_names = self.net.get_input_names(self.graph_name)
        self.output_names = self.net.get_output_names(self.graph_name)
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_names[0])
        self.batch_size = self.input_shape[0]


    def __call__(self, input_ids, attention_mask, token_type_ids):
        input_batch = input_ids.shape[0]
        processed_outputs = []
        if input_batch > self.batch_size:
            for start_idx in tqdm(range(0, input_batch, self.batch_size)):
                end_idx = min(start_idx + self.batch_size, input_batch)  # Ensure end_idx does not exceed input_batch
                input_ids_slice = input_ids[start_idx:end_idx]
                attention_mask_slice = attention_mask[start_idx:end_idx]
                token_type_ids_slice = token_type_ids[start_idx:end_idx]
                if input_ids_slice.shape[0] < self.batch_size:
                    padding_size = self.batch_size - input_ids_slice.shape[0]
                    input_ids_slice = np.concatenate([input_ids_slice, np.zeros((padding_size,) + input_ids_slice.shape[1:])], axis=0)
                    attention_mask_slice = np.concatenate([attention_mask_slice, np.zeros((padding_size,) + attention_mask_slice.shape[1:])], axis=0)
                    token_type_ids_slice = np.concatenate([token_type_ids_slice, np.zeros((padding_size,) + token_type_ids_slice.shape[1:])], axis=0)

                input_data = {self.input_names[0]: input_ids_slice,
                        self.input_names[1]: attention_mask_slice,
                        self.input_names[2]: token_type_ids_slice}
                results = self.net.process(self.graph_name, input_data)[self.output_names[0]]
                processed_outputs.append(results)
        else:
            padding_input_ids = None
            padding_attention_mask = None
            padding_token_type_ids = None
            if input_batch < self.batch_size:
                padding = np.zeros((self.batch_size - input_batch, *input_ids.shape[1:]))
                padding_input_ids = np.concatenate([input_ids, padding], axis=0)
                padding_attention_mask = np.concatenate([attention_mask, padding], axis=0)
                padding_token_type_ids = np.concatenate([token_type_ids, padding], axis=0)
            else:
                padding_input_ids = input_ids
                padding_attention_mask = attention_mask
                padding_token_type_ids = token_type_ids
            input_data = {self.input_names[0]: padding_input_ids,
                    self.input_names[1]: padding_attention_mask,
                    self.input_names[2]: padding_token_type_ids}
            results = self.net.process(self.graph_name, input_data)[self.output_names[0]]
            processed_outputs.append(results)

        return np.concatenate(processed_outputs, axis=0)[:input_batch]

