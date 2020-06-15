from objects_extractor import get_owners
from mask_generator import generate_masks
import enoki as ek
import mitsuba
import os
mitsuba.set_variant('gpu_autodiff_rgb')
from mitsuba.core import Thread
from mitsuba.core.xml import load_file
from mitsuba.python.util import traverse

# Absolute or relative path to the XML file
filename = 'scenes/cboxwithdragon/cboxwithdragon.xml'

# Add the scene directory to the FileResolver's search path
Thread.thread().file_resolver().append(os.path.dirname(filename))

# Load the scene
scene = load_file(filename)

# Find differentiable scene parameters
params = traverse(scene)
params.keep(['green.reflectance.value'])

owners = get_owners(params, scene)
masks = generate_masks(owners, scene)

from mitsuba.core import Color3f
green_ref = Color3f(params['green.reflectance.value'])
#light_ref = Color3f(params['light.reflectance.value'])

# Render a reference image (no derivatives used yet)
from mitsuba.python.autodiff import render, write_bitmap
image_ref = render(scene, spp=8)
crop_size = scene.sensors()[0].film().crop_size()
write_bitmap('renders/out_ref.png', image_ref, crop_size)
write_bitmap('renders/mask.png', masks['green'], crop_size)


# Change the left wall into a bright white surface
params['green.reflectance.value'] = [1.0,1.0,1.0]
params.update()

# Construct an Adam optimizer that will adjust the parameters 'params'
from mitsuba.python.autodiff import Adam
opt = Adam(params, lr=.2)

mask_len = ek.hsum(masks['green'])[0]

converged = False
it = 0
while converged != True and it <= 100:
    # Perform a differentiable rendering of the scene
    image = render(scene, optimizer=opt, unbiased=True, spp=1)

    #write_bitmap('render/out_%03i.png' % it, image, crop_size)

    # Objective: MSE between 'image' and 'image_ref'
    #ob_val = ek.hsum(ek.sqr(image - image_ref)) / len(image)
    ob_val = ek.hsum( masks['green'] * ek.sqr(image - image_ref)) / mask_len

    # Back-propagate errors to input parameters
    ek.backward(ob_val)

    # Optimizer: take a gradient step
    opt.step()

    err_ref = ek.hsum(ek.sqr(green_ref - params['green.reflectance.value']))
    #if err_ref[0] < 0.0001:
        #converged = True
    print('Iteration %03i : error= %g' % (it, err_ref[0]))
    it+=1
