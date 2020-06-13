import enoki as ek
import mitsuba
import os

mitsuba.set_variant('scalar_rgb')

from mitsuba.core import Thread
from mitsuba.core.xml import load_file
from mitsuba.python.util import traverse, ParameterMap

# Absolute or relative path to the XML file
filename = 'scenes/cboxwithdragon/cboxwithdragon.xml'

# Add the scene directory to the FileResolver's search path
Thread.thread().file_resolver().append(os.path.dirname(filename))

# Load the scene
scene = load_file(filename)

params = traverse(scene)

#List of owners of parameters, can be shapes, bsdfs, emitters and cameras
owners = list(set(map(lambda x: x.split('.')[0], params.keys())))

shapes = dict()
emitters = dict()

for o in owners:
    shapes[o] = list()
    for v in scene.shapes():
        if v.bsdf().id() == o:
            shapes[o].append(v)

print(shapes)




                

