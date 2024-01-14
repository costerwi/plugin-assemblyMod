from __future__ import print_function
from abaqusConstants import *

vp = session.viewports[session.currentViewportName]
ra = vp.displayedObject # should be in Assembly module
model = mdb.models[ra.modelName]

for surfName, surf in list(ra.surfaces.items()): # loop through all assembly surfaces
    if len(surf.edges):
        print('Skipping', surfName, 'which contains edges.')

    instSurf = None
    instFaces = {} # use to group faces by instance name

    if surf.faces: # made up of geometry faces
        if len(surf.faces) != len(surf.sides):
            print(surfName, 'number of faces does not match number of sides')
            continue

        for face, side in zip(surf.faces, surf.sides): # group by instance
            i = instFaces.setdefault(face.instanceName, {})
            i.setdefault(side, []).append(face) # group by side

    if surf.elements: # made up of element sides
        if len(surf.elements) != len(surf.sides):
            print(surfName, 'number of elements does not match number of sides')
            continue

        for elem, side in zip(surf.elements, surf.sides):
            i = instFaces.setdefault(elem.instanceName, {})
            i.setdefault(side, []).append(elem.label) # group by face number

    for instName, faces in list(instFaces.items()):
        inst = ra.instances[instName]
        if inst.dependent:
            inst = inst.part
        if surfName in list(inst.surfaces.keys()):
            print(surfName, 'already exists in', inst.name)
            continue

        elements = {}
        for side, faceList in list(faces.items()):
            if side in (FACE1, FACE2, FACE3, FACE4, FACE5, FACE6):
                # convert list of element ids to part MeshElementArray
                elements[side] = inst.elements.sequenceFromLabels(faceList)

        instSurf = inst.Surface(
                name=surfName,
                side1Faces=faces.get(SIDE1, []),
                side2Faces=faces.get(SIDE2, []),
                face1Elements=elements.get(FACE1, []),
                face2Elements=elements.get(FACE2, []),
                face3Elements=elements.get(FACE3, []),
                face4Elements=elements.get(FACE4, []),
                face5Elements=elements.get(FACE5, []),
                face6Elements=elements.get(FACE6, []),
                )

    if instSurf and 1 == len(instFaces): # this is a single-instance surface
        def replace(ref):
            for n in 'master', 'slave', 'surface', 'region':
                if not hasattr(ref, n):
                    continue
                region = getattr(ref, n)
                if not hasattr(region, '__len__') or 5 != len(region):
                    continue
                setName, owner, space, regionType, internal = region
                if 10 != regionType:
                    continue # not a surface
                if setName != surfName or 'Assembly' != owner:
                    continue
                keywords = {n: instSurf}
                ref.setValues(**keywords)

        # Search and replace old surface with new surface everywhere
        for load in list(model.loads.values()):
            replace(load)

        for constraint in list(model.constraints.values()):
            replace(constraint)

        del ra.surfaces[surfName]

