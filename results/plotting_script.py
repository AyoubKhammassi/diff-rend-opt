import matplotlib.pyplot as plt

#Parses the output file and returns a matrix with error values
def parse(resultsFile):
    
    results = open(resultsFile)
    resultLines = results.readlines()

    iteration = []
    error = []

    for line in resultLines:
        words = line.split(" ")
        iteration.append(int(words[1]))
        error.append(float(words[4].strip('\n')))
        
    results = [iteration, error]
    return results


classicResults = parse('ClassicResults.txt')
maskResults    = parse('MaskResults.txt')


plt.figure(figsize=(10,10))
plt.plot(classicResults[0], classicResults[1] ,label='classic results')
plt.plot(maskResults[0], maskResults[1] ,label='mask results')
plt.legend(loc="upper left")
#logarithmic scale
plt.yscale('log')


plt.ylabel('error')
plt.xlabel('iterations')
plt.show()