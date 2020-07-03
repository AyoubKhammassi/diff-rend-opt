'''
Benchmarking script to test the classic method and the masked one on the same scenes and parameters
For each test case we choose a scene and a parmater
We run the script once for use_masks = True and one for use_masks = False
We only use binary masks in these tests, the distance map is not tested yet
'''
import enoki as ek
import mitsuba
import os
import json
mitsuba.set_variant('gpu_autodiff_rgb')
from mitsuba.core import Thread
from mitsuba.core.xml import load_file
from mitsuba.python.util import traverse

# Absolute or relative path to the XML file
filename = 'scenes/cboxwithdragon/cboxwithdragon.xml'

#To easily switch between the existing method and the masked one
use_masks = True
base_path = 'results/EmitterSmall/'

if use_masks:
    base_path+='Masked/'
else:
    base_path+='Standard/'

#Where the rendered images from each iteration are going to be saved
dump_path = base_path+'iterations/'
try:
    os.makedirs(dump_path)
except:
    pass

#The parameters that we'll differentiate 
diff_param_key = 'emitter.radiance.value'
diff_param_owner = 'emitter'



# Add the scene directory to the FileResolver's search path
Thread.thread().file_resolver().append(os.path.dirname(filename))

# Load the scene
scene = load_file(filename)

# Find differentiable scene parameters
params = traverse(scene)
params.keep([diff_param_key])


from objects_extractor import get_owners
from mask_generator import generate_masks
#Get the parameters owner of the parameters that we'll differentiate
owners = get_owners(params, scene)
#Generate binary masks for those parameters owners
masks = generate_masks(owners, scene)


from mitsuba.core import Color3f
#Get a copy of the correct parameter value, this used later for the error calculation
diff_param_ref = Color3f(params[diff_param_key])

# Render a reference image to be used in the optimization loop
from mitsuba.python.autodiff import render, write_bitmap
image_ref = render(scene, spp=8)
crop_size = scene.sensors()[0].film().crop_size()
write_bitmap(base_path+'reference.png', image_ref, crop_size)


# Change the values of the parameters that we want to differentiate, and update the scene
params[diff_param_key] = [1.0,0.1,0.1]
params.update()

# Construct an SGD optimizer that will adjust the parameters 
from mitsuba.python.autodiff import SGD
opt = SGD(params, lr=.2, momentum=0.9)

#The sum of all weights in the mask is the used to normalize the loss function instead of the number of all pixels
mask_len = ek.hsum(masks[diff_param_owner])[0]
errors = list()
converged = False
it = 0
while converged != True and it <= 100:
    # Perform a differentiable rendering of the scene
    image = render(scene, optimizer=opt, unbiased=True, spp=1)

    #Dump this iteration's rendered image (Optional)
    write_bitmap(dump_path + 'out_%03i.png' % it, image, crop_size)

    #Use MSE or our modified version of it
    if use_masks:
        ob_val = ek.hsum( masks[diff_param_owner] * ek.sqr(image - image_ref)) / mask_len
    else:
        ob_val = ek.hsum(ek.sqr(image - image_ref)) / len(image)

    # Back-propagate errors to input parameters
    ek.backward(ob_val)

    # Optimizer: take a gradient step
    opt.step()

    # Calculate the error which is the difference between this iteration parameter value and the reference value
    err_ref = ek.hsum(ek.sqr(diff_param_ref - params[diff_param_key]))

    #Stop the loop when we reach a minimum error threshold (Optional)
    #if err_ref[0] < 0.0001:
        #converged = True
    #Add this error to the list of errors    
    errors.append(err_ref[0])
    print('Iteration %03i : error= %g' % (it, err_ref[0]))
    it+=1

#Dump the erros to a file, so we can use them later to compare the two methods
f = open(base_path+'Errors.txt', 'w')
json.dump(errors, f, indent=2)
f.close()
