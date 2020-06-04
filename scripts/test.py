import enoki as ek
import mitsuba
test = mitsuba.set_variant('gpu_autodiff_rgb')
from mitsuba.core import Thread
from mitsuba.core.xml import load_file
from mitsuba.python.util import traverse

# Load the Cornell Box
Thread.thread().file_resolver().append('dist/python/dragon')
scene = load_file('dist/python/dragon/dragon.xml')

# Find differentiable scene parameters
params = traverse(scene)
print(params)

params.keep(['red.reflectance.value','green.reflectance.value','light.reflectance.value'])

from mitsuba.core import Color3f
param_ref = Color3f(params['red.reflectance.value'])

print(param_ref)

# Render a reference image (no derivatives used yet)
from mitsuba.python.autodiff import render, write_bitmap
image_ref = render(scene, spp=8)
crop_size = scene.sensors()[0].film().crop_size()
write_bitmap('out_ref.png', image_ref, crop_size)


# Change the left wall into a bright white surface
params['red.reflectance.value'] = [.9, .9, .9]
params['green.reflectance.value'] = [.1, .1, .1]
params['light.reflectance.value'] = [.1,.1,.1]
params.update()

# Construct an Adam optimizer that will adjust the parameters 'params'
from mitsuba.python.autodiff import Adam
opt = Adam(params, lr=.2)

for it in range(100):
    # Perform a differentiable rendering of the scene
    image = render(scene, optimizer=opt, unbiased=True, spp=1)

    write_bitmap('out_%03i.png' % it, image, crop_size)

    # Objective: MSE between 'image' and 'image_ref'
    ob_val = ek.hsum(ek.sqr(image - image_ref)) / len(image)

    # Back-propagate errors to input parameters
    ek.backward(ob_val)

    # Optimizer: take a gradient step
    opt.step()

    err_ref = ek.hsum(ek.sqr(param_ref - params['red.reflectance.value']))
    print('Iteration %03i: error=%g' % (it, err_ref[0]))
