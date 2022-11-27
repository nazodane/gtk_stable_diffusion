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
import os

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
        global ImageFilter
        from PIL import Image, ImageFilter

        global autocast
        from torch import autocast
        from diffusers import DPMSolverMultistepScheduler
        global torch
        import torch
        try:
            from .lpw_stable_diffusion import StableDiffusionLongPromptWeightingPipeline
        except:
            from lpw_stable_diffusion import StableDiffusionLongPromptWeightingPipeline

#        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'garbage_collection_threshold:0.6,max_split_size_mb:50' #128
 
        global Path
        from pathlib import Path
        global home
        home = str(Path.home())
        global config_dir
        config_dir = home + "/.config/gtk-stable-diffusion/"
        global config_file_path
        config_file_path = config_dir + "config.toml"
        model_dir = home + "/.cache/huggingface/diffusers/sd-v1-4/"

# Note: We chose TOML because it's commentable (against JSON), simple (against YAML or XML), and non-ambiguous (against INI)
# Although we just implement toml dump as text dump because
# toml.load with toml.TomlPreserveCommentDecoder and toml.dump with toml.TomlPreserveCommentEncoder are completely broken.
        global dump_config
        def dump_config(conf):
            f_path = config_file_path
            toml_txt =  f"""
# nsfw_filter is for regulating erotics, grotesque, or ... something many normal things. [default=true]
# It's your responsibility to cater to your regulating authority wishes, not by us.
nsfw_filter = {"false" if "nsfw_filter" in conf and not conf["nsfw_filter"] else "true"}

# show_nsfw_filter_toggle is for you who don't want to change the nsfw toggle. [default=true]
show_nsfw_filter_toggle = {"false" if "show_nsfw_filter_toggle" in conf and not conf["show_nsfw_filter_toggle"] else "true"}
"""

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

#        repo_id = "/home/nazo/.cache/huggingface/diffusers/models--Deltaadams--Hentai-Diffusion/snapshots/8397ec1f41aeb904c9c3de8164fec8383abe0559/"
        repo_id = model_dir
        scheduler = DPMSolverMultistepScheduler.from_config(repo_id, subfolder="scheduler")
        pipe = StableDiffusionLongPromptWeightingPipeline.from_pretrained(repo_id, # revision="fp16",
            scheduler=scheduler)
        self.safety_checker = pipe.safety_checker
        pipe.safety_checker = None

        pipe = pipe.to("cuda")

        pipe.enable_xformers_memory_efficient_attention()
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        pipe.unet.to(memory_format=torch.channels_last)  # in-place operation

        torch.manual_seed(0)
        self.tensorsa = torch.tensor_split(torch.randn((1, 4, 512 // 8, 512 // 8), generator=None, device="cuda", dtype=torch.float).to(torch.float), 16, -1)
        self.pipe = pipe
        self.status_update('<big><b>Initializing: Done.</b></big>')
        self.delay_inited = True

    def process(self):
        prompt_buf = self.prompt_tv.get_buffer()
        prompt = prompt_buf.get_text(*prompt_buf.get_bounds(), True)
        neg_prompt_buf = self.neg_prompt_tv.get_buffer()
        neg_prompt = neg_prompt_buf.get_text(*neg_prompt_buf.get_bounds(), True)

        self.status_update('<big><b>Processing...</b></big>')
#        self._parent.debug_label.set_markup('<big><b>Prompt:</b> %s <b>Neg:</b> %s</big>'%(prompt, neg_prompt))
#        print("done -6")
        with autocast("cuda"):
#            print("done -5")
            torch.manual_seed(1)
#            print("done -4")
            tensorsa = self.tensorsa
#            print("done -3")
            tensorsb = torch.tensor_split(torch.randn((1, 4, 512 // 8, 512 // 8), generator=None, device="cuda", dtype=torch.float).to(torch.float), 16, -1)
#            print("done -2")

            # black magic
            latents = torch.cat((tensorsa[0], tensorsb[1], tensorsa[2], tensorsb[3], tensorsa[4], tensorsb[5], tensorsa[6], tensorsb[7],
                                 tensorsa[8], tensorsb[9], tensorsa[10], tensorsb[11], tensorsa[12], tensorsb[13], tensorsa[14], tensorsb[15]), axis=-1)
#            print("done -1")

            img_tensor = self.pipe(prompt, negative_prompt=neg_prompt, num_inference_steps=10, width=512, height=512, latents=latents, output_type="raw").images # [0]
#            print("done1")
            img_arr = img_tensor.cpu().float().numpy()
            if "nsfw_filter" not in self.conf or self.conf["nsfw_filter"] == True:
# copied and adopted from
# https://github.com/huggingface/diffusers/blob/2c6bc0f13ba2ba609ac141022b4b56b677d74943/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
                img = self.pipe.numpy_to_pil(img_arr)
                safety_checker_input = self.pipe.feature_extractor(img, return_tensors="pt").to("cuda")
                _, has_nsfw_concept = self.safety_checker(
                    images=img_arr, clip_input=safety_checker_input.pixel_values.to("cpu")
                )
                if has_nsfw_concept[0]:
                    print("NSFW")
                    img[0] = img[0].resize((16, 16), resample=Image.Resampling.BILINEAR)\
                                   .resize((512, 512), Image.Resampling.NEAREST)
                    img_arr = np.array(img[0]) / 255.0
                    img_arr = np.array([img_arr])
#            print("done2")
            img_ubarr = (img_arr * 255).round().astype("uint8")
#            print("done3")
            pixbuf = GdkPixbuf.Pixbuf.new_from_data(img_ubarr.flatten(), GdkPixbuf.Colorspace.RGB,
                                                        False, 8, 512, 512, 3*512)
#            print("done4")
            self.image.set_from_pixbuf(pixbuf)
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
            self.ls.append([pixbuf_prev, prompt, neg_prompt])
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
                try:
                    from .deep_danbooru_model import DeepDanbooruModel
                except:
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

        def new_prompt_textview(self, lm):
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
            tv._parent = self
            return tv
        self.prompt_tv = new_prompt_textview(self, lm)
        self.neg_prompt_tv = new_prompt_textview(self, lm)

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
            prompt_buf = self._parent.prompt_tv.get_buffer()
            prompt = prompt_buf.get_text(*prompt_buf.get_bounds(), True)
            neg_prompt_buf = self._parent.neg_prompt_tv.get_buffer()
            neg_prompt = neg_prompt_buf.get_text(*neg_prompt_buf.get_bounds(), True)
            self._parent.image.get_pixbuf().savev("%s||||%s.png"%(prompt,neg_prompt), "png") # XXX: more better naming rule?
            self._parent.debug_label.set_markup('<big><b>Saving: Done.</b></big>')

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



