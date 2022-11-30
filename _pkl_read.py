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
#with open(fpath, "rb") as f:
#    b = f.read()

#if b[0:4] != b"PK\x03\x04":
#    print("failed! not zip file")

#if b[8:10] != b"\x00\x00":
#    print("failed! the data is not store compression; the data may be compressed.") #



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

l = {}

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
        if i >= len(arr):
            break
        d[arr[i+1]] = arr[i]
#        print("%s = ./data/%s = data[%s:%s]"%(arr[i], arr[i+1], 0,0))
        i += 2

    return d

info_list = None
from zipfile import ZipFile
with ZipFile(fpath) as z: # 無圧縮ZIPリーダー
    info_list = z.infolist()

#import liburing
#ring = io_uring()
#cqes = io_uring_cqes()
#io_uring_queue_init(8, ring, 0) # 8???

data_dict = None
with open(fpath, "rb") as f:
    for i in info_list:
#        print(b[i.header_offset:i.header_offset+4]) # OK, now I got it!!!!
        if i.filename.endswith(".pkl"):
            f.seek(i.header_offset + 28)
            base = i.header_offset + 30 + len(i.filename) + \
                   int.from_bytes(f.read(2), byteorder="little")
            f.seek(base)
            pkl_buf = f.read(i.file_size)
            data_dict = pickle_to_datalist(pkl_buf)

    if not data_dict:
        print("Error: pickle is not found.")
        exit()

    for i in info_list:
        fn = i.filename.split("/")[-1]
        if fn in data_dict:
            f.seek(i.header_offset + 28)
            base = i.header_offset + 30 + len(i.filename) + \
                   int.from_bytes(f.read(2), byteorder="little")
#            f.seek(base)

            print("%s = ./data/%s = data[%s:%s]"%(data_dict[fn], fn, base, i.file_size))




# open
#_path = os.path.abspath(fpath).encode()
#sqe = io_uring_get_sqe(ring)  # sqe(submission queue entry)
#io_uring_prep_openat(sqe, -1, _path, os.O_RDONLY, 0o660)

# read
#buffer = bytearray(length)
#iov = iovec(buffer)

#io_uring_prep_read(sqe, fd, iov[0].iov_base, iov[0].iov_len, offset)
#read_length = _submit_and_wait(ring, cqes)  # get actual length of file read.


