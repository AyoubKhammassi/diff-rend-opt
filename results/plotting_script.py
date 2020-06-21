import matplotlib.pyplot as plt
import json


base_path = 'results/EmitterSmall/'
stdf = open(base_path+'Standard/Errors.txt', 'r')
maskedf = open(base_path+'Masked/Errors.txt', 'r')
stdResults = json.load(stdf)
maskResults = json.load(maskedf)


#plt.figure(figsize=(10,10))
plt.plot(stdResults)
plt.plot(maskResults)
#logarithmic scale
#plt.yscale('log')


plt.ylabel('error')
plt.xlabel('iteration')
#plt.show()
plt.savefig(base_path+'plot')