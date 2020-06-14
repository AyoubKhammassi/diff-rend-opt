import os
import enoki as ek
import numpy as np
import mitsuba

# Set the desired mitsuba variant
mitsuba.set_variant('gpu_autodiff_rgb')

from mitsuba.core import Float, UInt32, UInt64, Vector2f, Vector3f, Ray3f
from mitsuba.core import Bitmap, Struct, Thread
from mitsuba.core.xml import load_file, load_dict
from mitsuba.render import ImageBlock
from mitsuba.core.math import RayEpsilon, Infinity


'''def ray_intersect_mesh(mesh, rays):
    res = mesh.ray_intersect_triangle(0, rays)
    for i in range(mesh.face_count()):
        cur = mesh.ray_intersect_triangle(i, rays)
        mask = ~res[0] & cur[0]
        res[3][mask] = cur[3][mask]
        res[0][mask] = True
    return res'''

# Absolute or relative path to the XML file
filename = 'scenes/cboxwithdragon/cboxwithdragon.xml'

# Add the scene directory to the FileResolver's search path
Thread.thread().file_resolver().append(os.path.dirname(filename))

# Load the scene
scene = load_file(filename)
dragon = scene.shapes()[6]
shapes = dict()
shapes["type"] = "scene"
shapes["camera"] = scene.sensors()[0]
#shapes[dragon.id()] = dragon
shapes[scene.shapes()[4].id()] = scene.shapes()[4]


print(shapes)

dummy_scene = load_dict(shapes)


# Instead of calling the scene's integrator, we build our own small integrator
# This integrator simply computes the depth values per pixel
sensor = scene.sensors()[0]
film = sensor.film()
sampler = sensor.sampler()
film_size = film.crop_size()
spp = 3

# Seed the sampler
total_sample_count = ek.hprod(film_size) * spp

if sampler.wavefront_size() != total_sample_count:
    sampler.seed(UInt64.arange(total_sample_count))

# Enumerate discrete sample & pixel indices, and uniformly sample
# positions within each pixel.
pos = ek.arange(UInt32, total_sample_count)
pos //= spp
scale = Vector2f(1.0 / film_size[0], 1.0 / film_size[1])
pos = Vector2f(Float(pos  % int(film_size[0])),
               Float(pos // int(film_size[0])))

pos += sampler.next_2d()

# Sample rays starting from the camera sensor
rays, weights = sensor.sample_ray_differential(
    time=0,
    sample1=sampler.next_1d(),
    sample2=pos * scale,
    sample3=0
)

# Intersect rays with the scene geometry
surface_interaction = dummy_scene.ray_intersect(rays)

result = surface_interaction.t + 1.0
result[~(surface_interaction.t == Infinity)]=1
result[~surface_interaction.is_valid()] = 0

from mitsuba.python.autodiff import render, write_bitmap
crop_size = scene.sensors()[0].film().crop_size()
write_bitmap('renders/green_bin_mask.png', result, crop_size)
