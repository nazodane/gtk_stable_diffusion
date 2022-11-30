# Copyright 2022 Toshimitsu Kimura <lovesyao@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch

def pickle_to_datalist(d):
    import regex as re
# http://formats.kaitai.io/python_pickle/

#    import pickletools
#    import pipes
#    t = pipes.Template()
#    w = t.open("dis", "w")
#    arr = None
#    with t.open("dis", "r") as r:
#        pickletools.dis(d, out=w)
#        arr = re.findall(r"BINUNICODE '(.*)'", r.read())

#    "X" len str[len]
    arr = []
    for i, c in enumerate(d):
        if c == 0x58: # "X"
            n = d[i+5:i+5+int.from_bytes(d[i+1:i+5], byteorder="little")]
            try:
                n = n.decode("utf-8")
            except:
                continue
            arr += [n]

#            if re.fullmatch(r"^[a-zA-Z0-9._]+$", str(n)):
#                print(n)

    arr = [i for i in arr if i.find(".") > 0 or re.search(r'^\d+$', i)]

    i = 0
    d = {}
    while i < len(arr):
        while i < len(arr) and re.search(r'^\d+', arr[i]):
            arr.pop(0)
        if i+1 >= len(arr):
            break
        d[arr[i+1]] = arr[i]
#        print("%s = ./data/%s = data[%s:%s]"%(arr[i], arr[i+1], 0,0))
        i += 2

    return d

def pickle_data_read(fpath, read_list=None, write_to_tensor=False):
# fl = int.from_bytes(b[26:28], byteorder="little") # filename length
# el = int.from_bytes(b[28:30], byteorder="little") # extra field length
# dl = int.from_bytes(b[18:22], byteorder="little") # compressed data length -- 0???
# dl2 = int.from_bytes(b[22:26], byteorder="little") # original data length -- 0???
# hmm, but the file description flag is off...

# https://stackoverflow.com/questions/38437826/how-to-create-zip-file-with-data-descriptor-section
# https://gist.github.com/ysakasin/2edf8d3bf55c6ebf63f82851e302b030

# ./sd-v1-4/vae/diffusion_pytorch_model.bin 319.2MiB
# ./sd-v1-4/vae/diffusion_pytorch_model.bin.zst (zstd -9) 296.1MiB
# hmm, compression is not so useful.

    info_list = None
    from zipfile import ZipFile
    with ZipFile(fpath) as z: # read uncompressed Zip file
        info_list = z.infolist()

#import liburing
#ring = io_uring()
#cqes = io_uring_cqes()
#io_uring_queue_init(8, ring, 0) # 8???

    data_dict = None
    result_dict = {}
    with open(fpath, "rb") as f:
        for i in info_list:
#            print(b[i.header_offset:i.header_offset+4]) # OK, now I got it!!!!
            if i.filename.endswith(".pkl"):
                f.seek(i.header_offset + 28)
                base = i.header_offset + 30 + len(i.filename) + \
                       int.from_bytes(f.read(2), byteorder="little")
                f.seek(base)
                pkl_buf = f.read(i.file_size)
                data_dict = pickle_to_datalist(pkl_buf)

        if not data_dict:
            print("Error: pickle is not found.")
            return {}

        for i in info_list:
            fn = i.filename.split("/")[-1]
            if fn in data_dict:
                key = data_dict[fn]
                if read_list and key not in read_list:
                    continue
                f.seek(i.header_offset + 28)
                base = i.header_offset + 30 + len(i.filename) + \
                       int.from_bytes(f.read(2), byteorder="little")
                f.seek(base)
                if not write_to_tensor:
                    result_dict[key] = f.read(i.file_size)
                elif read_list:
                    tarr = read_list[key].numpy()
                    tarr_sz = tarr.size * tarr.itemsize

                    f.readinto(tarr) # XXX: overread on fp16 model...
                    if i.file_size == tarr_sz:
                        continue

#                    print("tensor size and file size is mismatching... (%s vs %s) at %s"%(i.file_size, tarr_sz, key))
                    if read_list[key].dtype == torch.float32 and i.file_size * 2 == tarr_sz:
#                        print("...assuming half float: ok")
                        read_list[key].data = torch.from_numpy(np.frombuffer(tarr, dtype=np.half)[0:tarr.size]).reshape(read_list[key].shape)
#                        result_dict = "half" # XXX
                        continue
                    if read_list[key].dtype == torch.int64 and i.file_size * 2 == tarr_sz: # used in RD1412.ckpt
#                        print("...assuming float: ok")
                        read_list[key].data = torch.from_numpy(np.frombuffer(tarr, dtype=np.float32)[0:tarr.size]).reshape(read_list[key].shape).long() # XXX: reallocate the buffer...
                        continue
                    if read_list[key].dtype == torch.int64 and i.file_size * 4 == tarr_sz: # used in trinart_characters_it4_v1.ckpt
#                        print("...assuming half float: ok")
                        read_list[key].data = torch.from_numpy(np.frombuffer(tarr, dtype=np.half)[0:tarr.size]).reshape(read_list[key].shape).long() # XXX: reallocate the buffer...
                        continue
                    print("tensor size and file size is mismatching... (%s vs %s:%s) at %s"%(i.file_size, tarr_sz, read_list[key].dtype, key))
                    #       (103680 vs 46080) at model.diffusion_model.input_blocks.0.0.weight @ sd-v1-5-inpainting

                else:
                    print("Error: write_to_tensor=True needs read_list")
                    return False
#                print("%s = ./data/%s = data[%s:%s]"%(data_dict[fn], fn, base, base+i.file_size))

## open
#    _path = os.path.abspath(fpath).encode()
#    sqe = io_uring_get_sqe(ring)  # submission queue entry
#    io_uring_prep_openat(sqe, -1, _path, os.O_RDONLY, 0o660)

## read
#    buffer = bytearray(length)
#    iov = iovec(buffer)

#    io_uring_prep_read(sqe, fd, iov[0].iov_base, iov[0].iov_len, offset)

## using cufile?
#    import kvikio
#    with kvikio.CuFile(path, "r") as f:
#        x = f.pread(buf, sz, file_offset=off)
#        x.get()

    return result_dict


# pickle_data_read("/home/nazo/.cache/huggingface/diffusers/sd-v1-4/sd-v1-4.ckpt")

# print(pickle_data_read("/home/nazo/.cache/huggingface/diffusers/sd-v1-4/sd-v1-4.ckpt", \
#                       ["cond_stage_model.transformer.text_model.encoder.layers.3.layer_norm2.weight",
#                        "cond_stage_model.transformer.text_model.encoder.layers.3.mlp.fc2.bias"]))

# import torch
# import numpy as np

# pkl = pickle_data_read("/home/nazo/.cache/huggingface/diffusers/sd-v1-4/sd-v1-4.ckpt", ["model.diffusion_model.time_embed.0.weight"])
# a =  torch.Tensor(np.frombuffer(pkl["model.diffusion_model.time_embed.0.weight"], dtype="float32")).reshape((1280, 320))
# print(a)
# sd = torch.load()["state_dict"]
# b = sd["model.diffusion_model.time_embed.0.weight"]
# print(a==b)

# a = torch.empty((1280, 320), dtype=torch.float32)
# pickle_data_read("/home/nazo/.cache/huggingface/diffusers/sd-v1-4/sd-v1-4.ckpt",\
#                 {"model.diffusion_model.time_embed.0.weight": a}, write_to_tensor = True)
# print(a)

