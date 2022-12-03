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

# Tested on Ubuntu 22.04 with Core i7-4790K DRAM 32GB / RTX 3060 VRAM 12GB
# Thank you if you send me enough amazon japan gift cards to my email to check other environment:P

import gi
gi.require_version("Gtk", "3.0")
gi.require_version('GtkSource', '3.0')
from gi.repository import Gtk, Gdk, Pango, GdkPixbuf, GtkSource, GLib
import threading
import os, sys

parent_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

class GTKStableDiffusion:
    def sd_init(self):
        # delayed load for faster start up!
        import faulthandler
        faulthandler.enable()
        global shutil
        import shutil

        global np
        import numpy as np
        global Image
        from PIL import Image

#        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'garbage_collection_threshold:0.6,max_split_size_mb:50' #128
 
        global Path
        from pathlib import Path
        global home
        home = str(Path.home())
        global config_dir
        config_dir = home + "/.config/gtk-stable-diffusion/"
        global config_file_path
        config_file_path = config_dir + "config.toml"

        global usable_models
        usable_models = {}

# original sd model check -> diffusers model check
        webui_dir = home + "/stable-diffusion-webui/models/Stable-diffusion/"
        if os.path.exists(webui_dir):
            for f in os.listdir(webui_dir):
                if f[-5:] == ".ckpt":
                    usable_models[f[:-5]] = webui_dir + f


        hf_df_dir = home + "/.cache/huggingface/diffusers/"
        if os.path.exists(hf_df_dir):
            for d in os.listdir(hf_df_dir): # auto search usable weights
                rev_path = hf_df_dir + d + "/refs/main"
                if not os.path.exists(rev_path):
                    repo_path = hf_df_dir + d + "/" # not standard but we used it

                    for f in os.listdir(repo_path):
                        if f[-5:] == ".ckpt": # something like the model before converting...
                            usable_models[f[:-5]] = repo_path + f

                    if os.path.exists(repo_path + "unet/diffusion_pytorch_model.bin"):
                        usable_models[d] = repo_path
                    continue
                with open(rev_path, "r") as f:
                    rev = f.read().replace("\n", "")
                repo_path = hf_df_dir + d + "/snapshots/" + rev + "/"
                if not os.path.exists(repo_path) or not os.path.exists(repo_path + "unet/diffusion_pytorch_model.bin"):
                    continue
                usable_models[d] = repo_path

        global secondary_model_to_string
        def secondary_model_to_string(model_dict):
            return "{" + (", ".join([ f'"%s" = "%s"'%(x,y) for x,y in model_dict.items()])) + "}"

# Note: We chose TOML because it's commentable (against JSON), simple (against YAML or XML), and non-ambiguous (against INI)
# Although we just implement toml dump as text dump because
# toml.load with toml.TomlPreserveCommentDecoder and toml.dump with toml.TomlPreserveCommentEncoder are completely broken.
        global dump_config
        def dump_config(conf):
            f_path = config_file_path
            _current_secondary_model = conf["current_secondary_model"] if "current_model" in conf and hasattr(conf["current_secondary_model"], "keys") else \
                                       {conf["current_secondary_model"]: "50%"} if "current_model" in conf and type(conf["current_secondary_model"]) == str else\
                                       {}
            _current_secondary_model_str = secondary_model_to_string(_current_secondary_model)
            print(_current_secondary_model_str)
            
            toml_txt =  f"""
# current_model is the current primary stable-diffusion weights for you to use. [default="sd-v1-4"]
current_model = "{conf["current_model"] if "current_model" in conf and conf["current_model"] in usable_models else "sd-v1-4"}"

# current_secondary_model is the current secondary stable-diffusion weights and percentages (a.k.a. model merging) for you to use. [default=""" + "{}" + f"""]
current_secondary_model = {_current_secondary_model_str}

# nsfw_filter is for regulating erotics, grotesque, or ... something many normal things. [default=true]
# It's your responsibility to cater to your regulating authority wishes, not by us.
nsfw_filter = {"false" if "nsfw_filter" in conf and not conf["nsfw_filter"] else "true"}

# show_nsfw_filter_toggle is for you who don't want to change the nsfw toggle. [default=true]
show_nsfw_filter_toggle = {"false" if "show_nsfw_filter_toggle" in conf and not conf["show_nsfw_filter_toggle"] else "true"}

# last_prompt is the last prompt that you use to generate. [default=""]
last_prompt = """ + '"""' + (conf["last_prompt"] if "last_prompt" in conf else "") + '"""'+ """

# last_neg_prompt is the last negative prompt that you use to generate. [default=""]
last_neg_prompt = """ + '"""' + (conf["last_neg_prompt"] if "last_neg_prompt" in conf else "") + '"""'+ """
""" # XXX: escapes are lacking...

            if os.path.exists(f_path):
                shutil.copy(config_file_path, config_file_path + ".bak") # save backup config
            with open(f_path, 'w') as f:
                f.write(toml_txt)

        if not os.path.exists(config_file_path):
            os.makedirs(config_dir, exist_ok=True)
            dump_config({}) # initialize config


        import toml
        try:
            self.conf = toml.load(config_file_path)
        except:
            shutil.copy(config_file_path, config_file_path + ".err")
            try:
                self.conf = toml.load(config_file_path+".bak") # read from backup config
            except:
                dump_config({}) # initialize config

        if not len(usable_models):
            import libtorrent as lt
            import time

            model_dir = home + "/.cache/huggingface/diffusers/sd-v1-4/"

            sess = lt.session({"enable_dht": True})

            # https://github.com/questianon/sdupdates/blob/main/sdupdates%20backup.md
            ml = "magnet:?xt=urn:btih:3A4A612D75ED088EA542ACAC52F9F45987488D1C&tr=udp://tracker.opentrackr.org:1337"

            mag = lt.parse_magnet_uri(ml)

            model_ckpt = model_dir + "sd-v1-4.ckpt"
            os.makedirs(model_dir, exist_ok=True)
            mag.save_path = model_dir

            hdl = sess.add_torrent(mag)

            self.status_update('<big><b>Initializing: Downloading: Metadata</b></big>')
            while (not hdl.status().has_metadata):
                time.sleep(1)
            self.status_update('<big><b>Initializing: Downloading: Data</b></big>')
            while (hdl.status().state != lt.torrent_status.seeding):
                s = hdl.status()

# begin
# https://gist.github.com/fabianp/1289286/08de70395530e28bfbb7c2ab1218a5a3d851e8f7
# Copyright Arvid Norberg 2008. Use, modification and distribution is
# subject to the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

                state_str = ['queued', 'checking', 'downloading metadata', \
                        'downloading', 'finished', 'seeding', 'allocating']
                self.status_update('<big><b>Initializing: Downloading: %s</b></big>'% ( \
                                  '%.2f%% complete (down: %.1f kb/s up: %.1f kB/s peers: %d) %s' % \
                                  (s.progress * 100, s.download_rate / 1000, s.upload_rate / 1000, \
                                  s.num_peers, state_str[s.state])))
                time.sleep(1)
# end

            os.system("python %s/convert_original_stable_diffusion_to_diffusers.py --checkpoint_path %s --dump_path %s"% \
                      (os.path.dirname(__file__), model_ckpt, model_dir))

            usable_models["sd-v1-4"] = model_dir

        global secondary_model_init
        def secondary_model_init(self):
            if "current_secondary_model" not in self.conf:
                if type(self.conf["current_secondary_model"]) == str:
                    self.conf["current_secondary_model"] = {self.conf["current_secondary_model"]: "50%"}
                if not hasattr(self.conf["current_secondary_model"], "keys"):
                    self.conf["current_secondary_model"] = {}
            global secondary_used
            secondary_used = 0
            for i, p in list(self.conf["current_secondary_model"].items()):
                if i not in usable_models:
                    del self.conf["current_secondary_model"][i]
                    continue
                secondary_used += int(p[0:-1])
        secondary_model_init(self)


        global save_prefix
        def save_prefix(self, prompt, postfix):
            import subprocess
            try:
                _ddir = subprocess.run(["xdg-user-dir", "DOWNLOAD"], capture_output=True, text=True).stdout.strip("\n")
            except FileNotFoundError:
                _ddir = home
            _ddir += "/gtk_stable_diffusion"

            if not os.path.exists(_ddir):
                os.makedirs(_ddir, exist_ok=True)

            prompt = prompt.replace("/", "_")
            fname_prefix = _ddir + "/" + " ".join(prompt.split(" ")[0:5]) + ("..." if len(prompt.split(" ")) > 5 else "") + "_"
            i = 0
            fname = fname_prefix + "0"
#            print(fname)
            while os.path.exists(fname + postfix): # postfix is something like ".png"
                 i += 1
                 fname = fname_prefix + str(i)
#                 print(fname)
            return fname

        print(self.conf["last_prompt"])
        prompt_buf = self.prompt_tv.get_buffer()
        prompt_buf.set_text(self.conf["last_prompt"])
        print(self.conf["last_neg_prompt"])
        neg_prompt_buf = self.neg_prompt_tv.get_buffer()
        neg_prompt_buf.set_text(self.conf["last_neg_prompt"])

        self.processing = True
        self.delay_inited = True
        self.process_modelload()

    def process_modelload(self):
        model_path = None
        model_id = ""
        model2_path = None
        model2_id = ""

        print(usable_models)

        if "current_model" in self.conf and self.conf["current_model"] in usable_models:
            model_id = self.conf["current_model"]
            model_path = usable_models[model_id]
            model_percentage = (100-secondary_used)/100.0

        model2_ids = self.conf["current_secondary_model"].keys()
        model2_paths = [usable_models[model2_id] for model2_id in model2_ids]
        model2_percentages = [int(v[0:-1])/100.0 for v in self.conf["current_secondary_model"].values()]
        print(model2_ids)
        print(model2_paths)
        print(model2_percentages)

        print("current primary model path: " + model_path)

        ms = model_id
        if len(model2_paths):
            print("current secondary model path: %s"%(model2_paths))
            ms = f"%s %s%%"%(model_id, 100-secondary_used)
            for k, v in self.conf["current_secondary_model"].items():
                ms += " / %s %s"%(k, v)

        self.status_update('<big><b>Model Loading (%s)...</b></big>'%(ms))

        import time
        time_pre = time.perf_counter()

        global torch
        import torch

        self.pipe = None
        torch.cuda.empty_cache()

        model_path = model_path
        pipe = None

        time_sta = time.perf_counter()
        time_mid1 = None
        time_mid2 = None

        from diffusers import DPMSolverMultistepScheduler
        from lpw_stable_diffusion import StableDiffusionLongPromptWeightingPipeline

        if model_path[-5:] == ".ckpt": # original sd model -- sd-v1-5-inpainting is not supoprted yet
            from transformers import CLIPTokenizer#, CLIPTextModel
#            import diffusers
            from ckpt_to_diffusers import ckpt_to_diffusers
            from free_weights import free_weights

            with torch.no_grad():
#                if hasattr(self, "pipe") and self.pipe:

#                    from ckpt_to_diffusers_read_list import ckpt_to_diffusers_read_list
#                    from pkl_read import pickle_data_read
#                    self.pipe = self.pipe
#                    # r = pickle_data_read(model_path, ckpt_to_diffusers_read_list(self.pipe.unet, self.pipe.vae, self.pipe.text_encoder), write_to_tensor=True, to_cuda=True)
#                    state_dict = pickle_data_read(model_path, ckpt_to_diffusers_read_list(self.pipe.unet, self.pipe.vae, self.pipe.text_encoder))
#                    ckpt_to_diffusers(state_dict, self.pipe.unet, self.pipe.vae, self.pipe.text_encoder)
#                    time_end = time.perf_counter()
#                    self.status_update('<big><b>Model Loading (%s): Done (%ss)</b></big>'%(ms, (time_end-time_sta)))
#                    self.processing = False
#                    return

#                    free_weights(self.pipe.unet, self.pipe.vae, self.pipe.text_encoder)
#                    torch.cuda.empty_cache()
#                    state_dict = torch.load(model_path, map_location="cuda:0")["state_dict"]
#                    ckpt_to_diffusers(state_dict, self.pipe.unet, self.pipe.vae, self.pipe.text_encoder)
#                    time_end = time.perf_counter()
#                    self.status_update('<big><b>Model Loading (%s): Done (%ss)</b></big>'%(ms, (time_end-time_sta)))
#                    self.processing = False
#                    return

# parameters are from https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml
                scheduler = DPMSolverMultistepScheduler(
                    beta_start=0.00085,
                    beta_end=0.012,
                    beta_schedule="scaled_linear",
                )

                # the initialization is so slow so we just use compiled one; see ../_gen_ckpt_base.py
                (unet, vae, text_model) = torch.load(os.path.dirname(__file__) + "/ckpt_base.pt")

                state_dict = torch.load(model_path, map_location="cuda:0")["state_dict"]

                for k, model2_path in enumerate(model2_paths):
                    if model2_path and model2_path[-5:] == ".ckpt":
                        state_dict2 = torch.load(model2_path, map_location="cpu")["state_dict"]
                        for i in state_dict: # on the fly model merging
                            if i.startswith("model.") and i in state_dict2:
                                if k == 0:
                                    state_dict[i] = model_percentage * state_dict[i] + model2_percentages[k] * state_dict2[i].cuda()
                                else:
                                    state_dict[i] += model2_percentages[k] * state_dict2[i].cuda()
                        del state_dict2
                        torch.cuda.empty_cache()
                    elif model2_path:
                        pipe2 = StableDiffusionLongPromptWeightingPipeline.from_pretrained(model2_path, # revision="fp16",
                               scheduler=scheduler, safety_checker = None
                           )

                        from ckpt_to_diffusers_read_list import ckpt_to_diffusers_read_list
                        read_list2 = ckpt_to_diffusers_read_list(pipe2.unet, pipe2.vae, pipe2.text_encoder)
                        for ckpt in read_list2:
                            if ckpt.startswith("model.") and ckpt in state_dict:
                                if k == 0:
                                    state_dict[ckpt] = model_percentage * state_dict[ckpt].reshape(read_list2[ckpt].data.shape) + model2_percentages[k] * read_list2[ckpt].cuda()
                                else:
                                    state_dict[ckpt] += model2_percentages[k] * read_list2[ckpt].cuda()
                        del read_list2
                        del pipe2
                        torch.cuda.empty_cache()

                ckpt_to_diffusers(state_dict, unet, vae, text_model)

#                from ckpt_to_diffusers_read_list import ckpt_to_diffusers_read_list
#                from pkl_read import pickle_data_read
#                r = pickle_data_read(model_path, ckpt_to_diffusers_read_list(unet, vae, text_model), write_to_tensor=True)

                time_mid1 = time.perf_counter()

                tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

                time_mid2 = time.perf_counter()

                pipe = StableDiffusionLongPromptWeightingPipeline(
                    vae=vae,
                    text_encoder=text_model,
                    tokenizer=tokenizer,
                    unet=unet,
                    scheduler=scheduler,
                    safety_checker=None, # delay nsfw filter init for faster initialization!
                    feature_extractor=None
                )

#            free_weights(pipe.unet, pipe.vae, pipe.text_encoder)
#            torch.cuda.empty_cache()
#            state_dict = torch.load(model_path, map_location="cuda:0")["state_dict"]
#            ckpt_to_diffusers(state_dict, unet, vae, text_model)

        else:
            with torch.no_grad():
                scheduler = DPMSolverMultistepScheduler.from_config(model_path, subfolder="scheduler")
                time_mid1 = time.perf_counter()
                pipe = StableDiffusionLongPromptWeightingPipeline.from_pretrained(model_path, # revision="fp16",
                           scheduler=scheduler, safety_checker = None # delay nsfw filter init for faster initialization!
                       )
                time_mid2 = time.perf_counter()

                for k, model2_path in enumerate(model2_paths):
                    if model2_path and model2_path[-5:] == ".ckpt":
                        state_dict2 = torch.load(model2_path, map_location="cpu")["state_dict"]
                        from ckpt_to_diffusers_read_list import ckpt_to_diffusers_read_list
                        read_list = ckpt_to_diffusers_read_list(pipe.unet, pipe.vae, pipe.text_encoder)
                        for ckpt in read_list:
                            if ckpt.startswith("model.") and ckpt in state_dict2:
                                if k == 0:
                                    read_list[ckpt].data = model_percentage * read_list[ckpt].data + model2_percentages[k] * state_dict2[ckpt].reshape(read_list[ckpt].data.shape)
                                else:
                                    read_list[ckpt].data += model2_percentages[k] * state_dict2[ckpt].reshape(read_list[ckpt].data.shape)
                        del state_dict2
                        del read_list
                    elif model2_path:
                        pipe2 = StableDiffusionLongPromptWeightingPipeline.from_pretrained(model2_path, # revision="fp16",
                                   scheduler=scheduler, safety_checker = None
                               )

                        from ckpt_to_diffusers_read_list import ckpt_to_diffusers_read_list
                        read_list = ckpt_to_diffusers_read_list(pipe.unet, pipe.vae, pipe.text_encoder)
                        read_list2 = ckpt_to_diffusers_read_list(pipe2.unet, pipe2.vae, pipe2.text_encoder)
                        for ckpt in read_list:
                            if ckpt.startswith("model."):
                                if k == 0:
                                    read_list[ckpt].data = model_percentage * read_list[ckpt].data + model2_percentages[k] * read_list2[ckpt].data
                                else:
                                    read_list[ckpt].data += model2_percentages[k] * read_list2[ckpt].data
                        del read_list
                        del read_list2
                        del pipe2

        pipe = pipe.to("cuda")

        pipe.enable_xformers_memory_efficient_attention()
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        pipe.unet.to(memory_format=torch.channels_last)  # in-place operation

        torch.manual_seed(0)
        self.tensorsa = torch.tensor_split(torch.randn((1, 4, 512 // 8, 512 // 8), generator=None, device="cuda", dtype=torch.float).to(torch.float), 16, -1)
        self.pipe = pipe
        time_end = time.perf_counter()
        self.status_update('<big><b>Model Loading (%s): Done (%.2fs, %.2fs, %.2fs, %.2fs)</b></big>'%(ms, \
                           (time_sta-time_pre), (time_mid1-time_sta), (time_mid2-time_mid1), (time_end-time_mid2)))
        self.processing = False

    def process(self):
        prompt_buf = self.prompt_tv.get_buffer()
        prompt = prompt_buf.get_text(*prompt_buf.get_bounds(), True)
        neg_prompt_buf = self.neg_prompt_tv.get_buffer()
        neg_prompt = neg_prompt_buf.get_text(*neg_prompt_buf.get_bounds(), True)
        self.conf["last_prompt"] = prompt
        self.conf["last_neg_prompt"] = neg_prompt
        dump_config(self.conf)

        self.status_update('<big><b>Processing...</b></big>')
#        self._parent.debug_label.set_markup('<big><b>Prompt:</b> %s <b>Neg:</b> %s</big>'%(prompt, neg_prompt))
#        print("done -6")

        view_pixbuf = None
        fname = ""
        with torch.autocast("cuda"):
#            print("done -5")
            count = self.batch_max if hasattr(self, "batch_max") else 1
            for n in range(count):
                torch.manual_seed(n + 1)

#                print("done -4")
                tensorsa = self.tensorsa
#                print("done -3")
                tensorsb = torch.tensor_split(torch.randn((1, 4, 512 // 8, 512 // 8), generator=None, device="cuda", dtype=torch.float).to(torch.float), 16, -1)
#                print("done -2")

                # black magic
                latents = torch.cat((tensorsa[0], tensorsb[1], tensorsa[2], tensorsb[3], tensorsa[4], tensorsb[5], tensorsa[6], tensorsb[7],
                                     tensorsa[8], tensorsb[9], tensorsa[10], tensorsb[11], tensorsa[12], tensorsb[13], tensorsa[14], tensorsb[15]), axis=-1)
#                print("done -1")

                img_tensor = self.pipe(prompt, negative_prompt=neg_prompt, num_inference_steps=10, width=512, height=512, latents=latents, output_type="raw").images # [0]
#                print("done1")
                img_arr = img_tensor.cpu().float().numpy()
                if "nsfw_filter" not in self.conf or self.conf["nsfw_filter"] == True:
# copied and adopted from
# https://github.com/huggingface/diffusers/blob/2c6bc0f13ba2ba609ac141022b4b56b677d74943/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
                    self.status_update('<big><b>Processing NSFW filter...</b></big>')
                    if not hasattr(self, "safety_checker") or not self.safety_checker:
                        from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
                        from transformers import AutoFeatureExtractor
                        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker")
                        self.feature_extractor = AutoFeatureExtractor.from_pretrained("CompVis/stable-diffusion-safety-checker")
                    img = self.pipe.numpy_to_pil(img_arr)
                    safety_checker_input = self.feature_extractor(img, return_tensors="pt").to("cuda")
                    _, has_nsfw_concept = self.safety_checker(
                        images=img_arr, clip_input=safety_checker_input.pixel_values.to("cpu")
                    )
                    if has_nsfw_concept[0]:
                        print("NSFW")
                        img[0] = img[0].resize((16, 16), resample=Image.Resampling.BILINEAR)\
                                       .resize((512, 512), Image.Resampling.NEAREST)
                        img_arr = np.array(img[0]) / 255.0
                        img_arr = np.array([img_arr])
#                print("done2")
                img_ubarr = (img_arr * 255).round().astype("uint8")
#                print("done3")
                pixbuf = GdkPixbuf.Pixbuf.new_from_data(img_ubarr.flatten(), GdkPixbuf.Colorspace.RGB,
                                                            False, 8, 512, 512, 3*512)

                if hasattr(self, "batch_max"):
                    if not view_pixbuf:
                        fname = save_prefix(self, prompt, "/")
                        print(fname)
                        if not os.path.exists(fname):
                            os.makedirs(fname, exist_ok=True)
                        open(fname + "/prompt.txt", "w").write('prompt = """%s"""\nneg_prompt = """%s"""\nprimary_model="%s"\nsecondary_model=%s'%(\
                                                        prompt, neg_prompt, self.conf["current_model"], \
                                                        secondary_model_to_string(self.conf["current_secondary_model"]))) # actually toml
                    pixbuf.savev(fname + "/%s.png"%(n), "png")

                    import math
                    ml = math.sqrt(self.batch_max)
                    s = int(512//ml)
                    item_img = np.array(Image.fromarray(img_ubarr[0]).resize((s, s), Image.Resampling.LANCZOS))
                    item_pixbuf = GdkPixbuf.Pixbuf.new_from_data(item_img.flatten(), GdkPixbuf.Colorspace.RGB,
                                                                 False, 8, s, s, 3*s)
                    if not view_pixbuf:
                        view_pixbuf = GdkPixbuf.Pixbuf.new_from_data(np.zeros(img_ubarr.flatten().shape, dtype=img_ubarr.flatten().dtype),
                                                                     GdkPixbuf.Colorspace.RGB, False, 8, 512, 512, 3*512)
                    item_pixbuf.copy_area(0, 0, s, s, view_pixbuf, s * int(n%ml), s * int(n//ml))
                    # XXX: textencode should reuse, cancellable, jax integration
                    # automatic prompt saving, output format (metadata in the image)
                    self.image.set_from_pixbuf(view_pixbuf)

            if hasattr(self, "batch_max"):
                self.status_update('<big><b>Processing: Batching: Done. (on %s)</b></big>'%os.path.dirname(fname))
                del self.batch_max
                self.processing = False
                return

#            print("done4")
            self.image.set_from_pixbuf(pixbuf)
            self.image_prompt = prompt
            self.image_neg_prompt = neg_prompt
            self.image_primary_model = self.conf["current_model"]
            self.image_secondary_model = secondary_model_to_string(self.conf["current_secondary_model"])

#            print("done5")

            self.inspect_process(img_tensor)

#            print("done16")
# make preview
            if not self.preview_generate: # re-generate from history
                self.status_update('<big><b>Processing: Done.</b></big>')
                self.processing = False
                return
#            print("done17")

            self.status_update('<big><b>Processing: Preview Generating...</b></big>')
#            print("done18")
            img_prev_ubarr = np.array(Image.fromarray(img_ubarr[0]).resize((64, 64), Image.Resampling.LANCZOS))
            pixbuf_prev = GdkPixbuf.Pixbuf.new_from_data(img_prev_ubarr.flatten(), GdkPixbuf.Colorspace.RGB,
                                                        False, 8, 64, 64, 3*64)

#            print("done19")
            self.ls.append([pixbuf_prev, prompt, neg_prompt]) # XXX: should save model data...
#            print("done20")

#        print("done21")
        self.status_update('<big><b>Processing: Done.</b></big>')
#        print("done22")
        self.processing = False
#        print("done23")

    def inspect_process(self, img_tensor):
#        print("done6")
        self.status_update('<big><b>Processing: Inspecting...</b></big>')
#        print("done7")

#        import time
#        time_sta = time.perf_counter()

        with torch.no_grad():# , torch.autocast("cuda")
#            print("done8")
            if not self.traced_fn:
                from deep_danbooru_model import DeepDanbooruModel
                global model

                deep_danbooru_path = config_dir + 'model-resnet_custom_v3.pt'
                if not os.path.exists(deep_danbooru_path):
                    os.makedirs(config_dir, exist_ok=True)
                    self.status_update('<big><b>Processing: Inspecting: Downloading...</b></big>')
                    from urllib.request import urlretrieve
                    deepdanbooru_url = "https://github.com/AUTOMATIC1111/TorchDeepDanbooru/releases/download/v1/model-resnet_custom_v3.pt"
                    urlretrieve(deepdanbooru_url, deep_danbooru_path)

                model = DeepDanbooruModel()
                model.load_state_dict(torch.load(deep_danbooru_path))

                deep_danbooru_ts_path = config_dir + "deep_danbooru.pt"
                if not os.path.exists(deep_danbooru_ts_path):
                    model.eval()
                    model.half().cuda()
                    traced_fn = torch.jit.trace(model, example_inputs=[img_tensor])
                    torch.jit.save(traced_fn, deep_danbooru_ts_path)

                self.traced_fn = torch.jit.load(deep_danbooru_ts_path)

#            print("done9")
            y = self.traced_fn(img_tensor)
#            print("done10")

#            time_mid = time.perf_counter()

            y = y[0].detach()
            mask = y >= 0.02
            y = y[mask].cpu().numpy()
            y_idx = torch.nonzero(mask)
#            print("done11")

            self.ls2 = Gtk.ListStore(str, float)
#            print("done12")
            for i, p in enumerate(y):
                self.ls2.append([model.tags[y_idx[i]], p])
#            print("done13")

#            time_end1 = time.perf_counter()

        sorted_ls2 = Gtk.TreeModelSort.new_with_model(self.ls2)
        sorted_ls2.set_sort_column_id(1, Gtk.SortType.DESCENDING)
        self.tv.set_model(sorted_ls2)
#        print("done14")
#        time_end2 = time.perf_counter()

        self.status_update('<big><b>Processing: Inspecting: Done</b></big>')
#        print("done15")

    def __init__(self):
        self.delay_inited = False
        self.processing = False

        self.window = Gtk.Window()
        self.window.set_title('My Stable Diffusion UI')
        exitf = lambda w, d=None: Gtk.main_quit()
        self.window.connect('destroy_event', exitf)
        self.window.connect('delete_event', exitf)

        def tv_kpef(self, event):

            def up_or_down(pos_char_start, pos_char_end, neg_char_start, neg_char_end):
                buf = self.get_buffer()
                if buf.get_has_selection():
#                    buf_start, buf_end = buf.get_bounds()
                    start, end = buf.get_selection_bounds()

                    text = buf.get_text(start, end, True)
                    print(text)
                    if len(text) > 1 and text[0] == neg_char_start and text[-1] == neg_char_end:
                        pend = end.copy()
                        pend.backward_char()
                        buf.delete(pend, end)
                        start, end = buf.get_selection_bounds()
                        nstart = start.copy()
                        nstart.forward_char()
                        buf.delete(start, nstart)
                        return

                    pstart = start.copy()
                    sb = pstart.backward_char()
                    nend = end.copy()
                    eb = nend.forward_char()

#                    if sb and eb: # not works
                    if start != pstart and end != nend:
                        sc = buf.get_text(pstart, start, True)
                        ec = buf.get_text(end, nend, True)
#                        print("is %s:%s %s:%s?"%(sc,ec,neg_char_start,neg_char_end))
                        if (sc == neg_char_start and ec == neg_char_end):
                            buf.delete(end, nend)
                            start, end = buf.get_selection_bounds()
                            pstart = start.copy()
                            pstart.backward_char()
                            buf.delete(pstart, start)
                            return

                    if sb and eb:
                        sc = buf.get_text(pstart, start, True)
                        ec = buf.get_text(end, nend, True)
                        if (sc == neg_char_start and ec == neg_char_end):
                            buf.delete(end, nend)
                            start, end = buf.get_selection_bounds()
                            pstart = start.copy()
                            pstart.backward_char()
                            buf.delete(pstart, start)
                            return

                    buf.insert(end, pos_char_end)
                    start, end = buf.get_selection_bounds()
                    end.backward_char()
                    buf.select_range(start, end) # reset slection position after insert
                    buf.insert(start, pos_char_start)
                    # selection
                return

            if event.keyval == Gdk.KEY_Up and event.get_state() & Gdk.ModifierType.CONTROL_MASK:
#                print("ctrl+up")
                up_or_down("(", ")", "[", "]")
                return True
            if event.keyval == Gdk.KEY_Down and event.get_state() & Gdk.ModifierType.CONTROL_MASK:
#                print("ctrl+down")
                up_or_down( "[", "]", "(", ")")
                return True

            if event.keyval != Gdk.KEY_Return:
                return False
            if self._parent.processing or not self._parent.delay_inited:
                return True # newline is dropped

            self._parent.processing = True
            self._parent.preview_generate = True
            threading.Thread(target=self._parent.process).start()

            return True
            
        def iv_activef(self, path):
            if self._parent.processing or not self._parent.delay_inited:
                return True # XXX: is this ok?

            model = self.get_model()

            prompt_buf = self._parent.prompt_tv.get_buffer()
            prompt_buf.set_text(model[path][1])
            neg_prompt_buf = self._parent.neg_prompt_tv.get_buffer()
            neg_prompt_buf.set_text(model[path][2])

            self._parent.processing = True
            self._parent.preview_generate = False
            threading.Thread(target=self._parent.process).start()

            return True

#        lang_file_path = os.path.dirname(__file__) + "sd_prompt.lang"
        lm = GtkSource.LanguageManager()
        lm.set_search_path(lm.get_search_path()+[os.path.dirname(__file__)])

        def tv_completion(self):
            menu = Gtk.Menu()
            if not hasattr(self, "nltk_inited"):
                import nltk
                global wordnet
                from nltk.corpus import wordnet
                if not nltk.find("corpora/wordnet.zip"):
                    nltk.download('wordnet')
                if not nltk.find("corpora/omw-1.4.zip"):
                    nltk.download('omw-1.4')
                self.nltk_inited = True
            import itertools
            target_words = ""
            prompt_buf = self._parent.prompt_tv.get_buffer()
            target_words = prompt_buf.get_text(*prompt_buf.get_bounds(), True)

            import regex as re
            target_words = re.sub(r"""["'()+-_/\\|]""", " ", target_words)
            print(target_words)
            print(target_words.split(" "))

            candidates = []
            if self._id == "prompt":
                candidates =  [[x.lemma_names() for x in  wordnet.synsets(w)] for w in target_words.split(" ")]
            elif self._id == "neg_prompt":
                candidates = [[[[k.name() for k in i.antonyms()] for i in x.lemmas()] for x in  wordnet.synsets(w)] for w in target_words.split(" ")]

            def flatten_and_uniq_str(arr):
                return {i for t in [[i] if isinstance(i, str) else flatten_and_uniq_str(i) for i in arr] for i in t}

            def on_prompt_add(self):
                self._tv.get_buffer().insert_at_cursor(self.get_label())

            candidates = flatten_and_uniq_str(candidates)
            for candidate in candidates:
                cd_menu = Gtk.MenuItem.new_with_label(candidate)
                cd_menu._tv = self
                menu.append(cd_menu)
                cd_menu.connect("activate", on_prompt_add)
                cd_menu.show()
            menu.popup(None, None, None, None, 0, Gtk.get_current_event_time())

        def new_prompt_textview(self, lm, _id):
            buf = GtkSource.Buffer()
#            print(lm.get_language_ids())
            buf.set_language(lm.get_language("sd_prompt"))
            tv = GtkSource.View(buffer=buf)
            tv.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
            tv.set_accepts_tab(False)
            # get font height
            fontsize = -1
            pg_ctx = tv.get_pango_context()
            if pg_ctx:
                fontdesc = pg_ctx.get_font_description()
                fontsize = fontdesc.get_size() / Pango.SCALE;

            tv.set_size_request(-1, fontsize * 5) # show 5 lines
            tv.connect('key-press-event', tv_kpef)
            tv.connect('show-completion', tv_completion)
            tv._parent = self
            tv._id = _id
            return tv
        self.prompt_tv = new_prompt_textview(self, lm, "prompt")
        self.neg_prompt_tv = new_prompt_textview(self, lm, "neg_prompt")

        self.debug_label = Gtk.Label()
        self.debug_label.set_markup('<big><b>Initializing...</b></big>')
        self.status_update = lambda x: GLib.idle_add(lambda self:self.debug_label.set_markup(x), self)

        self.vbox = Gtk.VBox()

        def wrap_frame(wid):
            sw = Gtk.ScrolledWindow()
            sw.set_hexpand( True )
            sw.set_border_width( 3 )
            sw.add( wid )
            frame = Gtk.Frame()
            frame.add( sw )
            return frame

        self.image_hbox = Gtk.HBox()

        self.image = Gtk.Image()
        self.image.set_pixel_size(512)

        def on_save(self):
            self._parent.debug_label.set_markup('<big><b>Saving...</b></big>')

            prompt = self._parent.image_prompt
            fname = save_prefix(self._parent, prompt, ".png")

            open(fname + ".txt", "w").write('prompt = """%s"""\nneg_prompt = """%s"""\nprimary_model="%s"\nsecondary_model=%s'%(\
                                            prompt, self._parent.image_neg_prompt, \
                                            self._parent.image_primary_model, self._parent.image_secondary_model)) # actually toml
            self._parent.image.get_pixbuf().savev(fname + ".png", "png") # XXX: we should save on metadata?
            self._parent.debug_label.set_markup('<big><b>Saving: Done. (on %s)</b></big>'%(os.path.dirname(fname)))

        def on_model_change(self):
            model_id = self.get_label()

            if model_id in self._parent.conf["current_secondary_model"]:
                del self._parent.conf["current_secondary_model"][model_id]
                secondary_model_init(self._parent)

            self._parent.conf["current_model"] = model_id
            dump_config(self._parent.conf)
            self._parent.processing = True
            threading.Thread(target=self._parent.process_modelload).start()

        def on_model2_change(self):
            model_id = self._model_id
            percentage = self.get_label()

            if percentage == "None":
                if model_id in self._parent.conf["current_secondary_model"]:
                    del self._parent.conf["current_secondary_model"][model_id]
            else:
                self._parent.conf["current_secondary_model"][model_id] = percentage
            secondary_model_init(self._parent)
            dump_config(self._parent.conf)
            
            self._parent.processing = True
            threading.Thread(target=self._parent.process_modelload).start()

        def on_bg(self):
            self._parent.processing = True
            self._parent.batch_max = self._count
            threading.Thread(target=self._parent.process).start()

        def show_menu(self, event):
            if event.type != Gdk.EventType.BUTTON_PRESS or event.button != 3 or not self._parent.delay_inited:
                return
            menu = Gtk.Menu()
            menu_count = 0
            if "show_nsfw_filter_toggle" in self._parent.conf and self._parent.conf["show_nsfw_filter_toggle"]:
                nsfwf_menu = Gtk.CheckMenuItem.new_with_label("NSFW Filter OFF")
                nsfwf_menu._parent = self._parent
                nsfwf_menu.set_active(True if "nsfw_filter" in self._parent.conf and not self._parent.conf["nsfw_filter"] else False)
                menu.append(nsfwf_menu)
                def on_nsfwf_toggle(self):
                    self._parent.conf["nsfw_filter"] = not self.get_active()
                    dump_config(self._parent.conf)
                nsfwf_menu.connect("toggled", on_nsfwf_toggle)
                nsfwf_menu.show()
                menu_count += 1

            if len(usable_models) > 1: # TODO: and not self._parent.processing
                model_menu = Gtk.MenuItem.new_with_label("Current Primary Model")
                model_menu_child = Gtk.Menu()
                model_menu.set_submenu(model_menu_child)
                for m_name in usable_models:
                    mmenu = Gtk.CheckMenuItem.new_with_label(m_name)
                    mmenu.set_active(True if "current_model" in self._parent.conf and m_name == self._parent.conf["current_model"] else False)
                    mmenu._parent = self._parent
                    mmenu.connect("activate", on_model_change)
                    model_menu_child.append(mmenu)
                    mmenu.show()
                menu.append(model_menu)
                model_menu.show()
                menu_count += 1

            if len(usable_models) > 1: # TODO: and not self._parent.processing
                model2_menu = Gtk.MenuItem.new_with_label("Current Secondary Model")
                model2_menu_child = Gtk.Menu()
                model2_menu.set_submenu(model2_menu_child)

                m2menu = Gtk.CheckMenuItem.new_with_label("None")
                m2menu.set_active(True if "current_secondary_model" not in self._parent.conf or self._parent.conf["current_secondary_model"]=="None" else False)
                m2menu._parent = self._parent
#                m2menu.connect("activate", on_model2_change)

                for m_name in usable_models:
                    if "current_model" in self._parent.conf and m_name == self._parent.conf["current_model"]:
                        continue
                    m2menu = Gtk.CheckMenuItem.new_with_label(m_name)
                    m2menu_child = Gtk.Menu()
                    m2menu.set_submenu(m2menu_child)
                    m2menu.set_active(True if "current_secondary_model" in self._parent.conf and m_name in self._parent.conf["current_secondary_model"] else False)
                    m2menu._parent = self._parent
                    m2menu.connect("select", lambda self: \
                        self.set_active(False if "current_secondary_model" in self._parent.conf and\
                            self.get_label() in self._parent.conf["current_secondary_model"] else True)) # XXX: for prevent bugs on select
#                    m2menu.connect("activate", on_model2_change)


                    val = int(self._parent.conf["current_secondary_model"][m_name][0:-1]) if m_name in self._parent.conf["current_secondary_model"] else 0
                    for p in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
                        if (p >= 100 - secondary_used + val): # don't over 100%
                            break
                        pmenu = Gtk.CheckMenuItem.new_with_label(str(p) + "%" if p else "None")
                        pmenu.set_active(True if p == val else False)
                        pmenu._model_id = m_name
                        pmenu._parent = self._parent
                        pmenu.connect("activate", on_model2_change)
                        m2menu_child.append(pmenu)
                        pmenu.show()
                    model2_menu_child.append(m2menu)
                    m2menu.show()                    
                menu.append(model2_menu)
                model2_menu.show()
                menu_count += 1

            if not self._parent.processing:
                batch_menu = Gtk.MenuItem.new_with_label("Batch Generate")
                batch_menu_child = Gtk.Menu()
                batch_menu.set_submenu(batch_menu_child)
                for batch_count in [4, 16, 64, 256]:
                    bgmenu = Gtk.MenuItem.new_with_label("%s"%(batch_count))
                    bgmenu._count = batch_count
                    bgmenu._parent = self._parent
                    bgmenu.connect("activate", on_bg)
                    batch_menu_child.append(bgmenu)
                    bgmenu.show()
                menu.append(batch_menu)
                batch_menu.show()
                menu_count += 1

            if len(self._parent.ls) > 0:
                save_menu = Gtk.MenuItem.new_with_label("Save")
                save_menu._parent = self._parent
                save_menu.connect("activate", on_save)
                menu.append(save_menu)
                save_menu.show()
                menu_count += 1

            if menu_count == 0:
                return
            menu.popup(None, None, None, None, 0, Gtk.get_current_event_time())

        self.eb = Gtk.EventBox()
        self.eb.add(self.image)
        self.eb._parent = self
        self.eb.connect("button-press-event", show_menu)
        self.image_hbox.pack_start(self.eb, False, False, 0)

        self.iv = Gtk.IconView.new()
        self.ls = Gtk.ListStore(GdkPixbuf.Pixbuf, str, str)
        self.iv.set_model(self.ls)
        self.iv.set_pixbuf_column(0)
        self.iv.set_text_column(1)
#        self.iv.set_column_spacing(0)
#        self.iv.set_item_padding(0)
        self.iv.set_item_width(64)
        self.iv.set_row_spacing(0)
        self.iv.set_spacing(0)
        self.iv.connect("item-activated", iv_activef)
#        self.iv.set_size_request(64*2, -1)
        self.iv._parent = self

        img_frame = wrap_frame(self.iv)
#        img_frame.set_size_request(64*2, -1) # XXX: size is not optimal

        self.nb = Gtk.Notebook()
        self.nb.set_tab_pos(Gtk.PositionType.TOP)

        self.tv = Gtk.TreeView()
        renderer = Gtk.CellRendererText()
        column = Gtk.TreeViewColumn("Tag", renderer, text=0)
        self.tv.append_column(column)
        column = Gtk.TreeViewColumn("Probability", renderer, text=1)
        self.tv.append_column(column)

#        self.ls2 = Gtk.ListStore(str, float)
#        self.tv.set_model(self.ls2)

        tv_frame = wrap_frame(self.tv)
        self.nb.append_page(tv_frame)
        self.nb.set_tab_label_text(tv_frame, "Inspect") # Inspect current image tags for negative prompt

        self.nb.append_page(img_frame)
        self.nb.set_tab_label_text(img_frame, "History")

#        def nb_switchf(self, widget, num):
#            if num != 0 or len(self._parent.ls) == 0 or not self._parent.delay_inited or self._parent.processing:
#                return
#            self._parent.processing = True
#            threading.Thread(target=self._parent.inspect_process).start()

        self.nb._parent = self
#        self.nb.connect("switch-page", nb_switchf)

        self.image_hbox.add(self.nb)

        self.vbox.add(self.image_hbox)

        self.prompt_hbox = Gtk.HBox()
        self.prompt_icon = Gtk.Image.new_from_icon_name("list-add", Gtk.IconSize.MENU)
        self.prompt_hbox.pack_start(self.prompt_icon, False, False, 0)
        self.prompt_hbox.add(wrap_frame(self.prompt_tv))
        self.vbox.add(self.prompt_hbox)

        self.neg_prompt_hbox = Gtk.HBox()
        self.neg_prompt_icon = Gtk.Image.new_from_icon_name("list-remove", Gtk.IconSize.MENU)
        self.neg_prompt_hbox.pack_start(self.neg_prompt_icon, False, False, 0)
        self.neg_prompt_hbox.add(wrap_frame(self.neg_prompt_tv))
        self.vbox.add(self.neg_prompt_hbox)

        self.vbox.add(self.debug_label)
        # gtk.ProgressBar()

        self.window.add(self.vbox)
        self.window.show_all()

# TODO: auto save?

        self.traced_fn = None
        threading.Thread(target=self.sd_init).start()

def main():
    hw = GTKStableDiffusion()
    Gtk.main()

if __name__ == "__main__": 
    main() 



