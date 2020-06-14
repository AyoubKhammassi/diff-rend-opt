import enoki as ek
import mitsuba

# Set the desired mitsuba variant
mitsuba.set_variant('gpu_autodiff_rgb')

from mitsuba.core import Float, UInt32, UInt64, Vector2f, Vector3f, Ray3f
from mitsuba.core.xml import load_dict
from mitsuba.core.math import Infinity


def generate_masks(owners, scene):
    #Generate rays that are used for both the binary mask and the distance mask
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

    #Masks dictionary containas a mask for each params
    masks = dict()
    #For each parameter owner, we generate a mask
    for key in owners:
        scene_dict = dict()
        scene_dict["type"] = "scene"
        scene_dict["camera"] = scene.sensors()[0]
        for item in owners[key]:
            scene_dict[item.id()] = item
        #dummy_scene contains only camera and objects for associated with this owner
        dummy_scene = load_dict(scene_dict)
        surface_interaction = dummy_scene.ray_intersect(rays)

        result = surface_interaction.t
        result[~(surface_interaction.t == Infinity)]=1
        result[~surface_interaction.is_valid()] = 0
        masks[key] = result
    return masks