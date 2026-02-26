"""Code to decode input data from file streams.

The code in this file is adapted from the torchdata and webdatasets packages.
It implements an extension based decoder, which selects a different
decoding function based on the extension.  In contrast to the
implementations in torchdata and webdatasets, the extension is
removed from the data field after decoding.  This ideally makes
the output format invariant to the exact decoding strategy.

Example:
    image.jpg will be decoded into a numpy array and will be accessable in the field `image`.
    image.npy.gz will be decoded into a numpy array which can also be accessed under `image`.

"""
import gzip
import json
import os
import pickle
from io import BytesIO
from typing import Any, Callable, Dict, Optional

import numpy
import torch
from torch.utils.data.datapipes.utils.decoder import imagespecs
from torchdata.datapipes.utils import StreamWrapper

# MODIFICATION ###########################
import re

MOVIA_LABEL_PREFIX = ["size_label", "color_label", "material_label", "shape_label"]

MOVIA_LABEL_ID2WORD = {
    "color_label": {
        0: "red", 1: "blue", 2: "green", 3: "yellow",
        4: "orange", 5: "purple", 6: "pink", 7: "brown",
    },
    "material_label": {0: "wood", 1: "plastic"},
    "shape_label": {0: "cube", 1: "circle", 2: "square"},
    "size_label": {0: "small", 1: "large"},
}

def basic_tokenize(text: str):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\sA-Za-z\d]", text.lower())

with open("/scratch/wz3008/new_SlotAttn/slot_attn_new/vocab.json") as f:
    VOCAB = json.load(f)

SOS = VOCAB["<sos>"]
EOS = VOCAB["<eos>"]
PAD = VOCAB["<pad>"]
UNK = VOCAB["<unk>"]
MAX_SEQ_LEN = 75

def tokenize_one(text:str): # Tuple[List[int], int]
    # doc = self.nlp(cap)
    # words = [t.text for t in doc]
    words = basic_tokenize(text)  # e.g. ["red","wood","cube","small"]
    words = words[: MAX_SEQ_LEN - 2]
    
    ids = [SOS] + [VOCAB.get(w, UNK) for w in words] + [EOS]
    length = len(ids)
    
    if len(ids) < MAX_SEQ_LEN:
        ids = ids + [PAD] * (MAX_SEQ_LEN - len(ids))
    else:
        ids = ids[: MAX_SEQ_LEN]
        length = min(length, MAX_SEQ_LEN)

    return ids, length

###########################################

class ExtensionBasedDecoder:
    """Decode key/data based on extension using a list of handlers.

    The input fields are assumed to be instances of
    [StreamWrapper][torchdata.datapipes.utils.StreamWrapper],
    which wrap an underlying file like object.
    """

    def __init__(self, *handler: Callable[[str, StreamWrapper], Optional[Any]]):
        self.handlers = list(handler) if handler else []

    def decode1(self, name, data):
        if not data:
            return data

        new_name, extension = os.path.splitext(name)
        if not extension:
            return name, data

        for f in self.handlers:
            result = f(extension, data)
            if result is not None:
                # Remove decoded part of name.
                data = result
                name = new_name
                # Try to decode next part of name.
                new_name, extension = os.path.splitext(name)
                if extension == "":
                    # Stop decoding if there are no further extensions to be handled.
                    break
        return name, data

    def decode(self, data: dict):
        result = {}

        if data is not None:
            for k, v in data.items():
                if k[0] == "_":
                    if isinstance(v, StreamWrapper):
                        data_bytes = v.file_obj.read()
                        v.autoclose()
                        v = data_bytes
                    if isinstance(v, bytes):
                        v = v.decode("utf-8")
                        result[k] = v
                        continue
                decoded_key, decoded_data = self.decode1(k, v)
                result[decoded_key] = decoded_data
        return result

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Decode input dictionary."""
        return self.decode(data)


def basic_handlers(extension: str, data: StreamWrapper):

    if extension in {".txt", ".text", ".transcript"}:
        data_bytes = data.file_obj.read()
        data.autoclose()
        return data_bytes.decode("utf-8")

    if extension in {".cls", ".cls2", ".class", ".count", ".index", ".inx", ".id"}:
        data_bytes = data.file_obj.read()
        data.autoclose()
        try:
            return int(data_bytes)
        except ValueError:
            return None

    if extension in {".json", ".jsn"}:
        output = json.load(data.file_obj)
        data.autoclose()
        return output

    if extension in {".pyd", ".pickle"}:
        output = pickle.load(data.file_obj)
        data.autoclose()
        return output

    if extension == ".pt":
        output = torch.load(data.file_obj)
        data.autoclose()
        return output

    if extension in {".npy"}:
        with BytesIO(data.file_obj.read()) as f:
            data.autoclose()
            output = numpy.load(f, allow_pickle=False)
        return output
        
    # MODIFICATOIN: Materialize .npz contents here to avoid returning 
    # a lazy NpzFile that may reference a closed stream later.
    if extension in {".npz"}:
        with BytesIO(data.file_obj.read()) as f:
            data.autoclose()
            npz = numpy.load(f, allow_pickle=False)
            
            labels = {}
            for field in MOVIA_LABEL_PREFIX:
                if field in npz.files:
                    arr = npz[field]
                    if isinstance(arr, numpy.ndarray) and arr.dtype == numpy.uint16:
                        arr = arr.astype(numpy.int32)
                    labels[field] = arr
            
            n = None
            for field, arr in labels.items():
                n = int(arr.shape[0])
                break
            
            label_texts = []
            label_tokens = []
            
            
            sentence = "There is "
            if n is not None:
                for i in range(n):
                    words = ["a"]
                    for field in MOVIA_LABEL_PREFIX:
                        if field not in labels: continue
                        idx = int(labels[field][i])
                        w = MOVIA_LABEL_ID2WORD.get(field, {}).get(idx, None)
                        if w is not None:
                            words.append(w)
                    text = " ".join(words)  # e.g. "a red wood cube small"
                    sentence = sentence + text + ", "
            sentence = sentence + "in the picture."
            
            ids, ln = tokenize_one(sentence)
            
            out = {
                #**labels,
                "tok_ids": numpy.asarray(ids, dtype=numpy.int64),   # (L,)
                "tok_lns": numpy.int64(ln),                         # scalar
            }
            return out
            #return {k: npz[k] for k in npz.files}
    return None


def compression_handler(extension: str, data: StreamWrapper):
    if extension not in [".gzip", ".gz"]:
        return None

    return StreamWrapper(gzip.GzipFile(fileobj=data.file_obj), parent_stream=data)


class ImageHandler:
    def __init__(self, imagespec):
        assert imagespec in list(imagespecs.keys()), "unknown image specification: {}".format(
            imagespec
        )
        self.imagespec = imagespecs[imagespec.lower()]

    def __call__(self, extension: str, data: StreamWrapper):
        if extension.lower() not in {".jpg", ".jpeg", ".png", ".ppm", ".pgm", ".pbm", ".pnm"}:
            return None

        try:
            import numpy as np
        except ImportError as e:
            del e
            raise ModuleNotFoundError(
                "Package `numpy` is required to be installed for default image decoder."
                "Please use `pip install numpy` to install the package"
            )

        try:
            import PIL.Image
        except ImportError as e:
            del e
            raise ModuleNotFoundError(
                "Package `PIL` is required to be installed for default image decoder."
                "Please use `pip install Pillow` to install the package"
            )

        atype, etype, mode = self.imagespec
        img = PIL.Image.open(data.file_obj)
        # TODO: This could be a problem, check if we run into issue with StreamWrapper.
        img.load()
        data.autoclose()
        img = img.convert(mode.upper())
        if atype == "pil":
            return img
        elif atype == "numpy":
            result = np.asarray(img)
            assert (
                result.dtype == np.uint8
            ), "numpy image array should be type uint8, but got {}".format(result.dtype)
            if etype == "uint8":
                return result
            else:
                return result.astype("f") / 255.0
        elif atype == "torch":
            result = np.asarray(img)
            assert (
                result.dtype == np.uint8
            ), "numpy image array should be type uint8, but got {}".format(result.dtype)

            if etype == "uint8":
                result = np.array(result.transpose(2, 0, 1))
                return torch.tensor(result)
            else:
                result = np.array(result.transpose(2, 0, 1))
                return torch.tensor(result) / 255.0
        return None


default_image_handler = ImageHandler("rgb8")

default_decoder = ExtensionBasedDecoder(compression_handler, default_image_handler, basic_handlers)
