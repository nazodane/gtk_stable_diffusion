The GTK Stable Diffusion is a very simple text-to-image tool. The tool is based on [GTK](https://en.wikipedia.org/wiki/GTK) UI framework and [Diffusers](https://github.com/huggingface/diffusers)' [Stable Diffusion](https://en.wikipedia.org/wiki/Stable_Diffusion) text-to-image pipeline.

GTK Stable Diffusion aims just for fun. This means the tool is more focusing on ~~spells~~ prompts than many small adjustments and random generations.

Installation
============
GTK Stable Diffusion is easily installable via pip command:
```bash
pip install gtk_stable_diffusion
```
or
```bash
pip install git+https://github.com/nazodane/gtk_stable_diffusion.git
```

Note that the model weights are automatically downloaded via [BitTorrent magnet link](https://en.wikipedia.org/wiki/Magnet_URI_scheme) using [libtorrent](https://en.wikipedia.org/wiki/Libtorrent) and the model weights are automatically converted from original Stable Diffusion format to Diffusers format in the first launch of the tool.

Usage
=====
```bash
~/.local/bin/gtk-stable-diffusion
```

Note: ctrl+space will show the candidates of synonyms for prompt and antonyms for negative prompt


Requirements
============
* Linux
* Python 3.10 or later
* CUDA 11.7 or later
* DRAM 16GB or higher
* RTX 3060 12GB or higher (the VRAM usage is currently over 8GB!)

Recommendations
===============
* Ubuntu 22.04 or later
* DRAM 32GB or higher
* NVMe SSD
* Faster non-restricted internet connections

License
=======
GTK Stable Diffusion codes are under Apache License 2.0. This is because we almost depend on Diffusers.

GTK and [its Python bindings are LGPL](https://www.gtk.org/docs/language-bindings/python) so we should carefully treat GTK-related codes.

Screenshot
==========
![Screenshot Image](screenshot.png)


