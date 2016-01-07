"""Script to adjust Abaqus/CAE assembly structure.

Carl Osterwisch <costerwi@gmail.com>
December 2013
"""

def instance_delete(instances):
    " Called by Abaqus/CAE plugin to remove selected instances "
    from abaqus import session
    vp = session.viewports[session.currentViewportName]
    ra = vp.displayedObject
    print("{}/{} instances selected for deletion.".format(
        len(instances), len(ra.instances)))
    vp.disableRefresh()
    vp.disableColorCodeUpdates()
    for i in instances:
        del ra.features[i.name]
    vp.enableColorCodeUpdates()
    vp.enableRefresh()


def instance_delete_shell():
    """Delete instances that have zero volume."""
    from abaqus import session
    vp = session.viewports[session.currentViewportName]
    ra = vp.displayedObject
    remove = []
    for inst in ra.instances.values():
        if not inst.part.getVolume():
            remove.append(inst)
    for inst in remove:
        del ra.instances[inst.name]
    print("{} empty instances removed.".format(len(remove)))


def instance_matchname():
    " Called by Abaqus/CAE plugin to rename instances based on part names "
    from math import log10
    from abaqus import session
    vp = session.viewports[session.currentViewportName]
    ra = vp.displayedObject

    parts = {}  # Identify unique parts and their instances
    for inst in ra.instances.values():
        parts.setdefault(inst.partName, []).append(inst.name)
    print("{} unique parts in {} instances.".format(
        len(parts), len(ra.instances)))

    newNames = {}   # Dict of new names to fix Loads, BCs, interations, etc.
    for partName, instNames in parts.items():
        if 1 == len(instNames):
            instName = instNames[0]
            toName = newNames.setdefault(instName, partName)    # no number necessary
            ra.features.changeKey(fromName=instName, toName=toName)
        else:
            fmt = "%s-%%0%dd"%(partName, 1 + int(log10(len(instNames))))
            for n, instName in enumerate(instNames):    # number each instance
                toName = newNames.setdefault(instName, fmt%(n + 1))
                ra.features.changeKey(fromName=instName, toName=toName)
    # TODO: seek out and fix Loads, BCs, interactions, etc.


def part_deleteUnused():
    " Called by Abaqus/CAE plugin to remove unused parts "
    from abaqus import session, mdb
    vp = session.viewports[session.currentViewportName]
    ra = vp.displayedObject
    model = mdb.models[ra.modelName]
    parts = set(model.parts.keys())
    used = set([inst.partName for inst in ra.instances.values()])
    unused = parts - used
    print("{} unused parts out of {} total to be deleted.".format(
        len(unused), len(parts)))
    for partName in unused:
        del model.parts[partName]


def part_derefDuplicate():
    " Replace repeated parts with one part "
    from numpy import asarray, allclose
    from abaqus import session
    from abaqusConstants import HIGH

    vp = session.viewports[session.currentViewportName]
    ra = vp.displayedObject

    similarMass = {}
    for inst in ra.instances.values():
        part = inst.part
        if part.name in similarMass:
            continue
        massProp = part.getMassProperties(
                relativeAccuracy=HIGH,
                specifyDensity=True, density=1)
        mass = massProp['mass']
        if mass:
            # Group parts by approximate mass
            similarMass.setdefault(int(round(mass)),
                    {}).setdefault(part.name, (part, massProp))

    vp.disableColorCodeUpdates()
    count = 0
    for similarParts in similarMass.values():
        # Dict of parts with similar mass
        while len(similarParts) > 1:
            name, (masterPart, masterProp) = similarParts.popitem()
            masterMoment = masterProp['momentOfInertia']
            masterCentroid = asarray(masterProp['centerOfMass'])
            unmatched = {}
            for name, (slavePart, slaveProp) in similarParts.items():
                slaveMoment = slaveProp['momentOfInertia']
                if not allclose(slaveMoment, masterMoment,
                        atol=1e-6*max(abs(asarray(slaveMoment)))):
                    # TODO Check for rotated instances
                    unmatched.setdefault( name, (slavePart, slaveProp) )
                    continue
                slaveCentroid = asarray(slaveProp['centerOfMass'])

                # replace all instances of this slavePart
                for inst in ra.instances.values():
                    if not inst.part == slavePart:
                        continue
                    inst.replace(masterPart)
                    inst.translate(slaveCentroid - masterCentroid
                            + inst.getTranslation())
                    count += 1
            similarParts = unmatched
    vp.enableColorCodeUpdates()
    print("{} instances updated.".format(count))
