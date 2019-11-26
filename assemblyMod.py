"""Script to adjust Abaqus/CAE assembly structure.

Carl Osterwisch <costerwi@gmail.com> December 2013
vim:foldmethod=indent
"""

def instance_editPart(instance):
    " Called by Abaqus/CAE plugin to edit parts "
    from abaqus import session
    vp = session.viewports[session.currentViewportName]
    ra = vp.displayedObject
    part = instance.part
    count = -1
    for inst in ra.instances.values():
        try:
            if inst.part == part:
                count += 1
        except AttributeError:
            continue
    print("instance {} references part {} used by {} other instances.".format(
        instance.name, part.name, count))
    vp.setValues(displayedObject=part)


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


def instance_delete_hollow():
    """Delete instances that have zero volume."""
    from abaqus import session
    vp = session.viewports[session.currentViewportName]
    ra = vp.displayedObject
    remove = []
    for inst in ra.instances.values():
        if ra.features[inst.name].isSuppressed():
            continue
        try:
            if 0 == len(inst.part.cells) or 0 == inst.part.getVolume():
                remove.append(inst)
        except AttributeError:
            continue
    vp.disableRefresh()
    for inst in remove:
        del ra.instances[inst.name]
    vp.enableRefresh()
    print("{} hollow instances removed.".format(len(remove)))


def instance_hideUnselected(instances):
    " Called by Abaqus/CAE plugin to hide unselected instances "
    from abaqus import session
    vp = session.viewports[session.currentViewportName]
    assembly = vp.displayedObject
    allNames = set(assembly.instances.keys())
    selectedNames = set(i.name for i in instances)
    hide = allNames - selectedNames
    print("Hiding {} instances.".format(len(hide)))
    vp.assemblyDisplay.hideInstances(instances=list(hide))


def instance_matchname():
    " Called by Abaqus/CAE plugin to rename instances based on part names "
    from math import log10
    from abaqus import session
    vp = session.viewports[session.currentViewportName]
    ra = vp.displayedObject

    parts = {}  # Identify unique parts and their instances
    for n, inst in enumerate(ra.instances.values()):
        if ra.features[inst.name].isSuppressed():
            continue
        tempName = 'temp~{}'.format(n)
        try:
            parts.setdefault(inst.partName, []).append(tempName)
        except AttributeError:
            continue
        vp.assemblyDisplay.showInstances(instances=(inst.name,))
        try:
            ra.features.changeKey(fromName=inst.name, toName=tempName)
        except ValueError as e:
            print("Warning: {!s}".format(e))
    print("{} unique parts in {} instances.".format(
        len(parts), len(ra.instances)))

    newNames = {}   # Dict of new names to fix Loads, BCs, interations, etc.
    for partName, instNames in parts.items():
        if 1 == len(instNames):
            instName = instNames[0]
            toName = newNames.setdefault(instName, partName)    # no number necessary
            try:
                ra.features.changeKey(fromName=instName, toName=toName)
            except ValueError as e:
                print("Warning: {!s}".format(e))
        else:
            nDigits = 1 + int(log10(len(instNames)))
            for n, instName in enumerate(sorted(instNames)):    # number each instance
                toName = newNames.setdefault(instName, "{0}-{1:0{2}}".format(
                    partName, n + 1, nDigits))
                try:
                    ra.features.changeKey(fromName=instName, toName=toName)
                except ValueError as e:
                    print("Warning: {!s}".format(e))
    # TODO: seek out and fix Loads, BCs, interactions, etc.


def part_deleteUnused():
    " Called by Abaqus/CAE plugin to remove unused parts "
    from abaqus import session, mdb
    vp = session.viewports[session.currentViewportName]
    ra = vp.displayedObject
    model = mdb.models[ra.modelName]
    parts = set(model.parts.keys())
    used = set()
    for inst in ra.instances.values():
        try:
            used.add(inst.partName)
        except AttributeError:
            continue
    unused = parts - used
    print("{} unused parts out of {} deleted.".format(
        len(unused), len(parts)))
    vp.disableColorCodeUpdates()
    for partName in unused:
        del model.parts[partName]
    vp.enableColorCodeUpdates()


def part_principalProperties():
    """Calculate and report principal mass properties"""
    from abaqus import session
    import numpy as np
    from abaqusConstants import HIGH, CARTESIAN

    vp = session.viewports[session.currentViewportName]
    part = vp.displayedObject

    massProp = part.getMassProperties(
            relativeAccuracy=HIGH,
            specifyDensity=True, density=1)
    vol = massProp['volume']
    if not vol:
        raise ZeroDivisionError('Part must have volume')
    mass = massProp['mass']
    print('{} mass {} (density {})'.format(part.name, mass, mass/vol))
    centroid = np.asarray(massProp['centerOfMass'])
    Ixx, Iyy, Izz, Ixy, Iyz, Ixz = massProp['momentOfInertia']

    A = np.array([[Ixx, Ixy, Ixz],
                  [Ixy, Iyy, Iyz],
                  [Ixz, Iyz, Izz]])
    evalues, evectors = np.linalg.eig(A)
    # evectors are column eigenvectors such evectors[:,i] corresponds to evalues[i]

    # Sort by eigenvalue so largest is first
    order = np.argsort(-evalues)
    Iz, Ix, Iy = evalues[order]
    if (Iz - Ix)/Iz < 0.01: # Iz is apporximately the same as Ix
        order = np.roll(order, 1) # Roll so that Ix and Iy are same
        Iz, Ix, Iy = evalues[order]
    rot = evectors[:, order]

    name = 'Principal csys'
    if part.features.has_key(name):
        del part.features[name]
    part.DatumCsysByThreePoints(
            name=name,
            coordSysType=CARTESIAN,
            origin=centroid,
            point1=centroid + rot[:, 1],
            point2=centroid + rot[:, 2],
        )
    print("\tIx={}, Iy={}, Iz={}".format(Ix, Iy, Iz))


def part_surfaceAreas():
    " Calculate area of all surfaces in the current part "
    from abaqus import session
    vp = session.viewports[session.currentViewportName]
    part = vp.displayedObject

    print(part.name)
    for surfName in part.surfaces.keys():
        surface = part.surfaces[surfName]
        print(surfName, part.getArea(surface.faces))


def part_derefDuplicate():
    " Replace repeated parts with one part "
    from numpy import log10, asarray, allclose
    from abaqus import session
    from abaqusConstants import HIGH

    vp = session.viewports[session.currentViewportName]
    ra = vp.displayedObject

    similarMass = {}
    for inst in ra.instances.values():
        if ra.features[inst.name].isSuppressed():
            continue
        try:
            part = inst.part
        except AttributeError:
            continue
        massProp = part.getMassProperties(
                relativeAccuracy=HIGH,
                specifyDensity=True, density=1)
        mass = massProp['mass']
        if mass:
            # Group parts by approximate mass
            similarMass.setdefault(int(round(log10(mass))),
                    {}).setdefault(part.name, (part, massProp))
    if len(similarMass) > 1:
        print("Found {} groups of parts with similar mass. Checking for duplicate parts within those groups.".format(len(similarMass)))

    vp.disableRefresh()
    vp.disableColorCodeUpdates()
    count = 0
    for similarParts in similarMass.values():
        # Dict of parts with similar mass
        while len(similarParts) > 1:
            name, (masterPart, masterProp) = similarParts.popitem()
            masterMoment = masterProp['momentOfInertia']
            masterCentroid = asarray(masterProp['centerOfMass'])
            masterArea = masterProp.get('area') or masterPart.getMassProperties(
                        regions=masterPart.faces,
                        relativeAccuracy=HIGH).get('area')
            unmatched = {} # Keep group of parts which do not match the current master
            for name, (slavePart, slaveProp) in similarParts.items():
                slaveMoment = asarray(slaveProp['momentOfInertia'])
                if not allclose(slaveMoment, masterMoment,
                        atol=1e-6*max(abs(slaveMoment))):
                    # TODO Use principal moment to check for rotated instances
                    unmatched.setdefault( name, (slavePart, slaveProp) )
                    continue
                slaveArea = slaveProp.get('area') or slavePart.getMassProperties(
                            regions=slavePart.faces,
                            relativeAccuracy=HIGH).get('area')
                if abs(masterArea - slaveArea)/masterArea > 0.01: # Surface area doesn't match
                    slaveProp['area'] = slaveArea
                    unmatched.setdefault( name, (slavePart, slaveProp) )
                    continue

                # Replace all Instances of this slavePart with the masterPart.
                # The difference in center of mass will be used to position the masterPart.
                slaveCentroid = asarray(slaveProp['centerOfMass'])
                for inst in ra.instances.values():
                    if ra.features[inst.name].isSuppressed():
                        continue
                    try:
                        if not inst.part == slavePart:
                            continue
                    except AttributeError:
                        continue
                    inst.replace(masterPart)
                    inst.translate(slaveCentroid - masterCentroid
                            + inst.getTranslation())
                    count += 1
            similarParts = unmatched # Continue to process any remaining parts
    vp.enableColorCodeUpdates()
    vp.enableRefresh()
    print("{} instances updated.".format(count))
