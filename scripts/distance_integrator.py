'''
Experimenting script for creating a distance map
This is useful for point emitters that don't cover any surface on the image
Instead of calculating a binary mask, we create a distance map from all the intersection points to the emitter's position
We normalize that distance and then we inverse it, that way the map stores non uniform weights, for how much each pixel is affected by the parameter changes of that emitter
'''
import os
import enoki as ek
import numpy as np
import mitsuba

# Set the desired mitsuba variant
mitsuba.set_variant('packet_rgb')

from mitsuba.core import Float, UInt32, UInt64, Vector2f, Vector3f
from mitsuba.core import Bitmap, Struct, Thread
from mitsuba.core.xml import load_file
from mitsuba.render import ImageBlock

# Absolute or relative path to the XML file
filename = 'scenes/cboxwithdragon/cboxwithdragon.xml'

# Add the scene directory to the FileResolver's search path
Thread.thread().file_resolver().append(os.path.dirname(filename))

# Load the scene
scene = load_file(filename)


# The custom integrator
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
surface_interaction = scene.ray_intersect(rays)

#We're using the dragon center as an example here
dragon_c = scene.shapes()[6].bbox().center()
c = Vector3f(dragon_c)
d = c - surface_interaction.p
result = ek.norm(d)
#Eliminate non valid intersection points
result[~surface_interaction.is_valid()] = 0

#We divide all distances by the maximum distance to normalize
max_dist = ek.hmax(result)
result /= max_dist
result = (1.0 - result)


from mitsuba.python.autodiff import write_bitmap
crop_size = scene.sensors()[0].film().crop_size()
write_bitmap('renders/green_dist_map.png', result, crop_size)
