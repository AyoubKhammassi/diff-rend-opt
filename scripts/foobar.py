from math import sqrt

def solution(area):
    remaining = area
    result = list()
    while remaining > 0:
        whole = int(sqrt(remaining))**(2)
        remaining -= whole
        result.append(whole)
    return result

