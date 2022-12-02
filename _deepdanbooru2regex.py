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

# python _deepdanbooru2regex.py  > sd_prompt.lang

import torch
from gtk_stable_diffusion import deep_danbooru_model

model = deep_danbooru_model.DeepDanbooruModel()
model.load_state_dict(torch.load('model-resnet_custom_v3.pt'))


header = """<?xml version="1.0" encoding="UTF-8"?>
<language id="sd_prompt" name="Stable Diffusion Prompt" version="2.0" _section="Others">
  <metadata>
    <property name="mimetypes">text/x-stable-diffusion-prompt</property>
    <property name="globs">*.sd_prompt</property>
  </metadata>

  <styles>
    <style id="tags1" name="Tags1" map-to="def:keyword"/>
    <style id="tags2" _name="Tags2" map-to="def:type"/>
<!--
    <style id="tags2" _name="Tags2" map-to="def:type"/>
    <style id="tags3" _name="Tags3" map-to="def:string"/>
    <style id="tags4" _name="Tags4" map-to="def:preprocessor"/>
    <style id="tags5" name="Tags5" map-to="def:special-char"/>
    <style id="tags6" _name="Tags6" map-to="def:comment"/>
    <style id="tags7" _name="Tags7" map-to="def:decimal"/>
    <style id="tags8" _name="Tags8" map-to="def:identifier"/>
    <style id="tags9" name="Tags5" map-to="def:floating-point"/>
    <style id="tags10" _name="Tags6" map-to="def:function"/>
    <style id="tags11" _name="Tags7" map-to="def:builtin"/>
    <style id="tags12" _name="Tags8" map-to="def:reserved"/>
-->

  </styles>

  <definitions>
    <context id="sd_prompt">
        <include>
"""
footer = """
        </include>
    </context>

  </definitions>
</language>
"""

print(header)

# romaji recognization
bo_on="aiueo"
shi_on="kstnhfmyrwgzdbpj"
shi2_on="tskysyshcychnyhymyrygyzybypy" # bhph
shi3_on="" # bhph

import regex as re

romajis = ["n"]

for s in bo_on:
    romajis += [s]
for s in shi_on:
    romajis += [s + b for b in bo_on] + [s + s + b for b in bo_on]
for i in range(len(shi2_on)//2):
    romajis += [shi2_on[i*2:i*2+2] + b for b in bo_on] + \
               [shi2_on[i*2] + shi2_on[i*2:i*2+2] + b for b in bo_on] # yossya

is_romaji = "(?:"+("|".join(romajis))+")+"

tags_raw = model.tags

def split_arr(tup):
    r = []
    for x in tup:
        r += [x[0:len(x)//2], x[len(x)//2:]]
    return r

# to avoid gtksource's pcre limitation
tags = []
for a in tags_raw:
#    tags += re.split('[_-]|\:|\(|\)|ing|er|ed|on', a)
    tags += re.split('[_-]|\:|\(|\)', a)
tags = list(dict.fromkeys(tags)) # deduplication

tags = [t for t in tags if not re.match("^"+is_romaji+"$", t)]
# print(len(tags))

#print(tags)

#tags = [a for a in tags if len(a)<9]

#[i for i in tags if len(i)>8]
# hmm...split the prefix and postfix?
# Counter([i[0:4] for i in tags]).most_common(1)
# co, han, shin, over, hoshi, under, sweat, cross, inter, sakura, breast, finger, person, express, antenna, project, feather, horizon, protect, mecha, etc...
# Counter([i[-3:] for i in tags]).most_common(1)
# ing, er, ed on, etc...

#tags = [a for a in tags_raw if len(a)<11]

arr = [tags]#split_arr(split_arr(split_arr([tags_raw])))

for idx, tags in enumerate(arr):
    for i in range(len(tags)):
        tags[i] = re.sub('([/!:<>=])', r'\\\1', re.escape(tags[i]))
    plane_list= " ".join(tags)

    #print(plane_regexp)
    with open("/tmp/regexp_py.txt", "w") as f:
        f.write(plane_list)

    # apt install libregexp-optimizer-perl
    perl_code = """
use Regexp::List;
my $o  = Regexp::List->new;
my $re = $o->list2re(qw/%s/);
print $re;
print "\\n";
"""%(plane_list)

# apt install libregexp-assemble-perl
#    perl_code = """
#use Regexp::Assemble;
# 
#my $ra = Regexp::Assemble->new;
#$ra->add( qr/%s/ );
#print $ra->re;
#print "\n";
#"""%(plane_regexp)

    with open("/tmp/regexp.pl", "w") as f:
        f.write(perl_code)

    import os
    os.system("perl /tmp/regexp.pl > /tmp/regexp.txt")

    with open("/tmp/regexp.txt", "r") as f:
        txt = f.read()

    print('<context id="tags%s" style-ref="tags%s">'%(idx+1,idx+1))
    print("<match>(?:^| |_|-|\\:|\\[|\\(|_\\\\\\()%s(?:\\\\?)(?:\\)|\\])?(?=_|-|\\:| |,|$)</match>"%(txt.replace("[/@_]", "[\\/@_]").replace("(?^:", "(?:").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace("\n","")))
#    print("<match>(^| |_|-|\\:|\\[|\\(|_\\\\\\()%s\\\\?(\\)|\\]|ing|er|ed|on)?</match>"%(txt.replace("[/@_]", "[\\/@_]").replace("(?^:", "(?:").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace("\n","")))
    print('</context>')

print('<context id="tags2" style-ref="tags2">')
print("<match>(?:^| |_|-|\\:|\\[|\\(|_\\\\\\()%s(?:\\\\?)(?:\\)|\\])?(?=_|-|\\:| |$)</match>"%(is_romaji))
print('</context>')

print(footer)
