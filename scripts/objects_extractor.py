
#returns a dictionary containing an entry for each parameter parent (BSDF, Emitter), the value is a list of objects
def get_owners(params, scene):
    owners_ids = list(set(map(lambda x: x.split('.')[0], params.keys())))

    #List of owners of parameters, can be shapes, bsdfs, emitters
    owners = dict()

    for o in owners_ids:
        owners[o] = list()
        for v in scene.shapes():
            if v.bsdf().id() == o:
                owners[o].append(v)
            if v.is_emitter() and v.emitter().id() == o:
                owners[o].append(v)
    return owners




                

