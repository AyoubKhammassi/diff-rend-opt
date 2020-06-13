import os
import enoki as ek
import numpy as np
import mitsuba

# Set the desired mitsuba variant
mitsuba.set_variant('gpu_autodiff_rgb')
import mitsuba.core

#from mitsuba.core import Float, UInt32, UInt64, Vector2f, Vector3f
from enoki.cuda_autodiff import UInt32, UInt64
from enoki.cuda_autodiff import Float32 as Float
from enoki.cuda_autodiff import Vector3f as Vector3fC
from enoki.cuda_autodiff import Vector2f as Vector2fC

from mitsuba.core import Bitmap, Struct, Thread, Vector2f, Vector3f, Ray3f, Object
from mitsuba.core.xml import load_file
from mitsuba.render import ImageBlock, PositionSample3f
from mitsuba.core.math import RayEpsilon

# Absolute or relative path to the XML file
filename = 'scenes/cboxwithdragon/cboxwithdragon.xml'

# Add the scene directory to the FileResolver's search path
Thread.thread().file_resolver().append(os.path.dirname(filename))

# Load the scene
scene = load_file(filename)

shapes = scene.shapes()
shapes = [s for s in shapes if s == scene.shapes()[6]]


# Instead of calling the scene's integrator, we build our own small integrator
# This integrator simply computes the depth values per pixel
sensor = scene.sensors()[0]
film = sensor.film()
sampler = sensor.sampler()
film_size = film.crop_size()
spp = 32

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
surface_interaction = scene.ray_intersect(rays)

dragon_c = scene.shapes()[6].bbox().center()
'''c = Vector3fC(dragon_c)
d = c - surface_interaction.p
result = ek.norm(d)
max_dist = ek.hmax(result)
result /= max_dist
result = 1.0 - result'''

rays = surface_interaction.spawn_ray_to(dragon_c)
rays.o += surface_interaction.n * RayEpsilon
#mask = scene.shapes()[6].ray_intersect_triangle(1,rays, surface_interaction.is_valid())

#result = mask[3]
surface_interaction = scene.ray_intersect(rays, surface_interaction.is_valid())
result = surface_interaction.t
result[~surface_interaction.is_valid()] = 0

# Given intersection, compute the final pixel values as the depth t
# of the sampled surface interaction
#surface_interaction = dragon.ray_intersect(rays)
#result = mask.t

# Set to zero if no intersection was found
#result[~mask[0]] = 0

block = ImageBlock(
    film.crop_size(),
    channel_count=5,
    filter=film.reconstruction_filter(),
    border=False
)
block.clear()
# ImageBlock expects RGB values (Array of size (n, 3))
block.put(pos, rays.wavelengths, Vector3f(result, result, result), 1)

# Write out the result from the ImageBlock
# Internally, ImageBlock stores values in XYZAW format
# (color XYZ, alpha value A and weight W)


xyzaw_np = np.array(block.data()).reshape([film_size[1], film_size[0], 5])

# We then create a Bitmap from these values and save it out as EXR file
bmp = Bitmap(xyzaw_np, Bitmap.PixelFormat.XYZAW)
bmp = bmp.convert(Bitmap.PixelFormat.RGBA, Struct.Type.Float32, srgb_gamma=False)
bmp.write('depth.exr')
