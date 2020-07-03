import matplotlib.pyplot as plt
import json

#For each test, we only change this path
base_path = 'results/EmitterSmall/'
stdf = open(base_path+'Standard/Errors.txt', 'r')
maskedf = open(base_path+'Masked/Errors.txt', 'r')

#Standard Method results
stdResults = json.load(stdf)
#Masked Method results
maskResults = json.load(maskedf)


plt.plot(stdResults)
plt.plot(maskResults)

#logarithmic scale
#plt.yscale('log')


plt.ylabel('error')
plt.xlabel('iteration')
#plt.show()
plt.savefig(base_path+'plot')