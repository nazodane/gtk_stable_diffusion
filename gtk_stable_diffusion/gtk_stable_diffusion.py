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
from gi.repository import Gtk, Gdk, Pango, GdkPixbuf, GtkSource
import threading

class GTKStableDiffusion:
    def sd_init(self):
        # delayed load for faster start up!
        global autocast
        from torch import autocast
        from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
        global torch
        import torch
#        import os
        global np
        import numpy as np
        global Image
        from PIL import Image

#        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'garbage_collection_threshold:0.6,max_split_size_mb:50' #128

        import os
        from pathlib import Path
        home = str(Path.home())
        model_dir = home + "/.cache/huggingface/diffusers/sd-v1-4/"

        model_dir_check = model_dir + "unet/diffusion_pytorch_model.bin"
        if not os.path.exists(model_dir_check):
            import libtorrent as lt
            import time

            sess = lt.session({"enable_dht": True})

            # https://github.com/questianon/sdupdates/blob/main/sdupdates%20backup.md
            ml = "magnet:?xt=urn:btih:3A4A612D75ED088EA542ACAC52F9F45987488D1C&tr=udp://tracker.opentrackr.org:1337"

            mag = lt.parse_magnet_uri(ml)

            model_ckpt = model_dir + "sd-v1-4.ckpt"
            os.makedirs(model_dir, exist_ok=True)
            mag.save_path = model_dir

            hdl = sess.add_torrent(mag)

            self.debug_label.set_markup('<big><b>Initializing: Downloading: Metadata</b></big>')
            while (not hdl.status().has_metadata):
                time.sleep(1)
            self.debug_label.set_markup('<big><b>Initializing: Downloading: Data</b></big>')
            while (hdl.status().state != lt.torrent_status.seeding):
                s = hdl.status()

# begin
# https://gist.github.com/fabianp/1289286/08de70395530e28bfbb7c2ab1218a5a3d851e8f7
# Copyright Arvid Norberg 2008. Use, modification and distribution is
# subject to the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

                state_str = ['queued', 'checking', 'downloading metadata', \
                        'downloading', 'finished', 'seeding', 'allocating']
                self.debug_label.set_markup('<big><b>Initializing: Downloading: %s</b></big>'% (
                                            '%.2f%% complete (down: %.1f kb/s up: %.1f kB/s peers: %d) %s' % \
                                            (s.progress * 100, s.download_rate / 1000, s.upload_rate / 1000, \
                                             s.num_peers, state_str[s.state])))
                time.sleep(1)
# end

            os.system("python %s/convert_original_stable_diffusion_to_diffusers.py --checkpoint_path %s --dump_path %s"% \
                      (os.path.dirname(__file__), model_ckpt, model_dir))

#        repo_id = "/home/nazo/.cache/huggingface/diffusers/models--Deltaadams--Hentai-Diffusion/snapshots/8397ec1f41aeb904c9c3de8164fec8383abe0559/"
        repo_id = model_dir
        scheduler = DPMSolverMultistepScheduler.from_config(repo_id, subfolder="scheduler")
        pipe = StableDiffusionPipeline.from_pretrained(repo_id, # revision="fp16",
           scheduler=scheduler) # safety_checker=None, 

        pipe = pipe.to("cuda")

        pipe.enable_xformers_memory_efficient_attention()
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        pipe.unet.to(memory_format=torch.channels_last)  # in-place operation

        torch.manual_seed(0)
        self.tensorsa = torch.tensor_split(torch.randn((1, 4, 512 // 8, 512 // 8), generator=None, device="cuda", dtype=torch.float).to(torch.float), 16, -1)
        self.pipe = pipe
        self.debug_label.set_markup('<big><b>Initializing: Done.</b></big>')
        self.delay_inited = True

    def process(self):
        prompt_buf = self.prompt_tv.get_buffer()
        prompt = prompt_buf.get_text(*prompt_buf.get_bounds(), True)
        neg_prompt_buf = self.neg_prompt_tv.get_buffer()
        neg_prompt = neg_prompt_buf.get_text(*neg_prompt_buf.get_bounds(), True)

        self.debug_label.set_markup('<big><b>Processing...</b></big>')
#        self._parent.debug_label.set_markup('<big><b>Prompt:</b> %s <b>Neg:</b> %s</big>'%(prompt, neg_prompt))

        with autocast("cuda"):
            for i in range(1):
                torch.manual_seed(i+1)
                tensorsa = self.tensorsa
                tensorsb = torch.tensor_split(torch.randn((1, 4, 512 // 8, 512 // 8), generator=None, device="cuda", dtype=torch.float).to(torch.float), 16, -1)

                # black magic
                latents = torch.cat((tensorsa[0], tensorsb[1], tensorsa[2], tensorsb[3], tensorsa[4], tensorsb[5], tensorsa[6], tensorsb[7],
                                     tensorsa[8], tensorsb[9], tensorsa[10], tensorsb[11], tensorsa[12], tensorsb[13], tensorsa[14], tensorsb[15]), axis=-1)

                image = self.pipe(prompt, negative_prompt=neg_prompt, num_inference_steps=10, width=512, height=512, latents=latents).images[0]
                arr = np.array(image).flatten()
                pixbuf = GdkPixbuf.Pixbuf.new_from_data(arr, GdkPixbuf.Colorspace.RGB,
                                                        False, 8, image.size[1], image.size[0], 3*image.size[1])
                self.image.set_from_pixbuf(pixbuf)

# make preview
                if not self.preview_generate: # re-generate from history
                    self.debug_label.set_markup('<big><b>Processing: Done.</b></big>')
                    self.processing = False
                    return
                img_prev = image.resize((64, 64), Image.Resampling.LANCZOS)
                arr_prev = np.array(img_prev).flatten()
                pixbuf_prev = GdkPixbuf.Pixbuf.new_from_data(arr_prev, GdkPixbuf.Colorspace.RGB,
                                                        False, 8, img_prev.size[1], img_prev.size[0], 3*img_prev.size[1])

                self.ls.append([pixbuf_prev, prompt, neg_prompt])

        self.debug_label.set_markup('<big><b>Processing: Done.</b></big>')
        self.processing = False

    def __init__(self):
        self.delay_inited = False
        self.processing = False

        self.window = Gtk.Window()
        self.window.set_title('My Stable Diffusion UI')
        exitf = lambda w, d=None: Gtk.main_quit()
        self.window.connect('destroy_event', exitf)
        self.window.connect('delete_event', exitf)

        def tv_kpef(self, event):
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

        def new_prompt_textview(self):
            buf = GtkSource.Buffer()
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
            tv._parent = self
            return tv
        self.prompt_tv = new_prompt_textview(self)
        self.neg_prompt_tv = new_prompt_textview(self)


        self.debug_label = Gtk.Label()
        self.debug_label.set_markup('<big><b>Initializing...</b></big>')

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

        self.image_hbox.pack_start(self.image, False, False, 0)

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
        img_frame.set_size_request(64*2, -1) # XXX: size is not optimal
        self.image_hbox.add(img_frame)

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

# TODO: save and auto save?

        threading.Thread(target=self.sd_init).start()

def main():
    hw = GTKStableDiffusion()
    Gtk.main()

if __name__ == "__main__": 
    main() 


