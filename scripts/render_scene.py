import enoki as ek
import mitsuba
import os

test = list()
test.append(10.00)
test.append(100.00)
test.append(259.00)
mitsuba.set_variant('gpu_autodiff_rgb')
from mitsuba.core import Thread
from mitsuba.core.xml import load_file
from mitsuba.python.util import traverse


# Absolute or relative path to the XML file
filename = 'scenes/cboxwithdragon/cboxwithdragon.xml'

dump_path = 'results/BSDFTest/'
try:
    os.makedirs(dump_path)
except:
    pass
# Add the scene directory to the FileResolver's search path
Thread.thread().file_resolver().append(os.path.dirname(filename))

# Load the scene
scene = load_file(filename)

# Render a reference image (no derivatives used yet)
from mitsuba.python.autodiff import render, write_bitmap
image_ref = render(scene, spp=8)

crop_size = scene.sensors()[0].film().crop_size()
write_bitmap(dump_path+'out_ref.png', image_ref, crop_size)