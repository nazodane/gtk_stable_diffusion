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


#fpath = "/home/nazo/.cache/huggingface/diffusers/models--hakurei--waifu-diffusion/snapshots/364c9bf63fea4cfd8908b6472308ad87c68137ea/text_encoder/pytorch_model.bin"
fpath = "/home/nazo/.cache/huggingface/diffusers/sd-v1-4/sd-v1-4.ckpt"
b = []
with open(fpath, "rb") as f:
    b = f.read()

if b[0:4] != b"PK\x03\x04":
    print("failed! not zip file")

if b[8:10] != b"\x00\x00":
    print("failed! the data is not store compression; the data may be compressed.") #

# /home/nazo/.cache/huggingface/diffusers/sd-v1-4/vae/diffusion_pytorch_model.bin 319.2MiB
# zstd -9 (diffusion_pytorch_model.bin.zst) 296.1MiB
# hmm, compression is not so much good.


# fl = int.from_bytes(b[26:28], byteorder="little")
# fn = b[30:30+fl] # file name

# el = int.from_bytes(b[28:30], byteorder="little") # extra field length
# dl = int.from_bytes(b[18:22], byteorder="little") # compressed data length -- 0???
# dl2 = int.from_bytes(b[22:26], byteorder="little") # original data length -- 0???
# hmm, but the file description flag is off...


# https://stackoverflow.com/questions/38437826/how-to-create-zip-file-with-data-descriptor-section
# https://gist.github.com/ysakasin/2edf8d3bf55c6ebf63f82851e302b030

pkl_buf = None
l = {}

from zipfile import ZipFile
with ZipFile(fpath) as z: # 無圧縮ZIPリーダー
    for i in z.infolist():
#        print(b[i.header_offset:i.header_offset+4]) # OK, now I got it!!!!
        base = i.header_offset + 30 + len(i.filename) + \
               int.from_bytes(b[i.header_offset + 28: i.header_offset + 30], byteorder="little")
        if i.filename.endswith(".pkl"):
            pkl_buf = b[base: base+i.file_size]
        else:
            l[i.filename.split("/")[-1]] = (base, base+i.file_size)

# http://formats.kaitai.io/python_pickle/

import pickletools
import pipes
import regex as re
t = pipes.Template()
w = t.open("dis", "w")
arr = None
with t.open("dis", "r") as r:
    pickletools.dis(pkl_buf, out=w)
    arr = re.findall(r"BINUNICODE '(.*)'", r.read())

arr = [i for i in arr if i.find(".") > 0 or re.search(r'^\d+$', i)]

i = 0
while i < len(arr):
    while re.search(r'^\d+$', arr[i]):
        i += 1
    print("%s = ./data/%s = data[%s:%s]"%(arr[i], arr[i+1], *l[arr[i+1]]))
    i += 2

