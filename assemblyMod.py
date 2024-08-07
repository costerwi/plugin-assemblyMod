"""Scripts to adjust Abaqus/CAE assembly structure.

Carl Osterwisch <costerwi@gmail.com> December 2013
"""

from __future__ import print_function
import os
import numpy as np
DEBUG = os.environ.get('DEBUG')

_principal_csys = 'Principal csys' # feature name

try:
    # Load Rotation class from scipy if it's available (CAE >= 2020)
    from scipy.spatial.transform import Rotation
except ImportError:
    from simpleTransform import Rotation

class ARotation(Rotation):
    """Extend Rotation class with some useful Abaqus CAE methods

    Carl Osterwisch, June 2022"""

    @classmethod
    def from_csys(cls, csys):
        """Define rotation based on DatumCsys orientation"""

        matrix = np.array([
            csys.axis1.direction,
            csys.axis2.direction,
            csys.axis3.direction,
        ]).T
        return cls.from_matrix(matrix)

    def as_axisAngle(self):
        """Convert to rotation around axis

        >>> axis = np.array([0.6, 0.8, 0])
        >>> angle = np.radians(90)
        >>> r = ARotation.from_rotvec(axis*angle)
        >>> axis2, angle2 = r.as_axisAngle()
        >>> np.allclose(axis, axis2)
        True
        >>> np.degrees(angle2)
        90.0
        """
        v = self.as_rotvec()
        theta = np.sqrt(v.dot(v))
        if not theta:
            return np.array([0., 0., 1.]), 0.
        return v/theta, theta

if hasattr(ARotation, 'as_dcm'):
    # as_dcm and from_dcm were renamed as_matrix and from_matrix in scipy 1.4 (CAE >2023)
    # Define the new names here for consistency
    ARotation.as_matrix = ARotation.as_dcm
    ARotation.from_matrix = ARotation.from_dcm


def humanDuration(seconds):
    """Represent seconds as appropriate seconds, minutes, hours, or days

    >>> humanDuration(155)
    '2.6 minutes'
    >>> humanDuration(33)
    '33 seconds'
    """
    if seconds > 1.1*86400:
        return '{:.2f} days'.format(seconds/86400.)
    if seconds > 1.1*3600:
        return '{:.1f} hours'.format(seconds/3600.)
    if seconds > 1.1*60:
        return '{:.1f} minutes'.format(seconds/60.)
    return '{:.2g} seconds'.format(seconds)


def statusGenerator(iterable, message='items', interval=20):
    """Iterate through iterable and print status messages every 'interval' seconds as needed"""
    from time import time
    nTotal = len(iterable)
    t0 = time() # starting time
    tUpdate = t0 + interval # seconds to first update
    for n, item in enumerate(iterable, 1):
        yield item
        t = time() # current time
        if n >= nTotal: # finished
            print('{} {} in {}.'.format(
                nTotal, message, humanDuration(t - t0)))
        elif t > tUpdate:
            dt = (nTotal - n)*(t - t0)/n # estimate seconds remaining
            print('{:.0f}% of {}. Estimated {} more to complete.'.format(
                100.0*n/nTotal, message, humanDuration(dt)))
            tUpdate = t + interval # seconds until next update

# {{{1 ASSEMBLY INSTANCES DELETE

def instance_delete(instances):
    " Called by Abaqus/CAE plugin to delete selected instances "
    from abaqus import session
    vp = session.viewports[session.currentViewportName]
    ra = vp.displayedObject
    print("{}/{} instances deleted.".format(
        len(instances), len(ra.instances)))
    ra.deleteFeatures([inst.name for inst in instances])


def instance_delete_hidden():
    """Delete part instances that are currently hidden."""
    from abaqus import session
    vp = session.viewports[session.currentViewportName]
    ra = vp.displayedObject
    displayed = set(vp.assemblyDisplay.visibleInstances) # set of names
    remove = []
    for inst in ra.instances.values():
        if not inst.name in displayed:
            remove.append(inst)
    instance_delete(remove)


def instance_delete_suppressed():
    """Delete part instances that are currently suppressed."""
    from abaqus import session
    vp = session.viewports[session.currentViewportName]
    ra = vp.displayedObject
    remove = []
    for inst in ra.instances.values():
        if not hasattr(inst, 'part'):
            continue # skip non-part instances
        if ra.features[inst.name].isSuppressed():
            remove.append(inst)
    instance_delete(remove)

# {{{1 ASSEMBLY INSTANCES SUPPRESS

def instance_suppress(instances):
    " Called by Abaqus/CAE plugin to delete selected instances "
    from abaqus import session
    vp = session.viewports[session.currentViewportName]
    ra = vp.displayedObject
    print("{} instances suppressed.".format(len(instances)))
    ra.suppressFeatures([inst.name for inst in instances])


def instance_suppress_part(instances):
    " Called by Abaqus/CAE plugin to suppress selected instances "
    from abaqus import session
    vp = session.viewports[session.currentViewportName]
    ra = vp.displayedObject
    parts = {inst.part.name for inst in instances if hasattr(inst, 'part')} # set of part names
    suppress = []
    for inst in ra.instances.values():
        if not hasattr(inst, 'part'):
            continue # skip non-part instances
        if ra.features[inst.name].isSuppressed():
            continue
        if inst.part.name in parts:
            suppress.append(inst)
    instance_suppress(suppress)


def instance_suppress_noArea():
    """Suppress instances that have zero area."""
    from abaqus import session
    vp = session.viewports[session.currentViewportName]
    ra = vp.displayedObject
    suppress = []
    for inst in ra.instances.values():
        if not hasattr(inst, 'part'):
            continue # skip non-part instances
        if ra.features[inst.name].isSuppressed():
            continue
        if 0 == len(inst.part.faces):
            suppress.append(inst)
    instance_suppress(suppress)


def instance_suppress_noVolume():
    """Suppress instances that have zero volume."""
    from abaqus import session
    vp = session.viewports[session.currentViewportName]
    ra = vp.displayedObject
    suppress = []
    for inst in ra.instances.values():
        if not hasattr(inst, 'part'):
            continue # skip non-part instances
        if ra.features[inst.name].isSuppressed():
            continue
        if 0 == len(inst.part.cells) and 0 == inst.part.getVolume():
            suppress.append(inst)
    instance_suppress(suppress)


def instance_suppress_unmeshed():
    """Suppress instances of parts which have unmeshed regions"""
    from abaqus import session
    vp = session.viewports[session.currentViewportName]
    ra = vp.displayedObject
    suppress = []
    for inst in ra.instances.values():
        if not hasattr(inst, 'part'):
            continue # skip non-part instances
        if ra.features[inst.name].isSuppressed():
            continue
        if inst.part.getUnmeshedRegions():
            suppress.append(inst)
    instance_suppress(suppress)


def instance_suppress_badElements():
    """Suppress instances of parts which have failed element analysis quality"""
    from abaqus import session
    from abaqusConstants import ANALYSIS_CHECKS
    vp = session.viewports[session.currentViewportName]
    ra = vp.displayedObject
    parts = {}  # partName => [ instances ]
    for inst in ra.instances.values():
        if not hasattr(inst, 'part'):
            continue # skip non-part instances
        if ra.features[inst.name].isSuppressed():
            continue
        parts.setdefault(inst.part.name, []).append(inst)
    suppress = []
    for partName, instances in statusGenerator(parts.items(), 'part meshes checked'):
        part = instances[0].part
        quality = part.verifyMeshQuality(ANALYSIS_CHECKS)
        if len(quality['failedElements']) > 0:
            suppress.extend(instances)
    instance_suppress(suppress)


def instance_suppress_invert():
    from abaqus import session
    vp = session.viewports[session.currentViewportName]
    ra = vp.displayedObject
    resume = []
    suppress = []
    for inst in ra.instances.values():
        if not hasattr(inst, 'part'):
            continue # skip non-part instances
        if ra.features[inst.name].isSuppressed():
            resume.append(inst.name)
        else:
            suppress.append(inst.name)
    ra.suppressFeatures(suppress)
    ra.resumeFeatures(resume)


def instance_suppress_resumeAll():
    from abaqus import session
    vp = session.viewports[session.currentViewportName]
    ra = vp.displayedObject
    resume = []
    for inst in ra.instances.values():
        if not hasattr(inst, 'part'):
            continue # skip non-part instances
        if ra.features[inst.name].isSuppressed():
            resume.append(inst.name)
    ra.resumeFeatures(resume)

# {{{1 ASSEMBLY INSTANCES HIDE

def instance_hide(instances):
    " Called by Abaqus/CAE plugin to hide selected instances "
    from abaqus import session
    vp = session.viewports[session.currentViewportName]
    ra = vp.displayedObject
    print("{} instances hidden.".format(len(instances)))
    vp.assemblyDisplay.hideInstances([inst.name for inst in instances])


def instance_hide_part(instances):
    " Called by Abaqus/CAE plugin to hide parts"
    from abaqus import session
    vp = session.viewports[session.currentViewportName]
    ra = vp.displayedObject
    parts = {inst.part.name for inst in instances if hasattr(inst, 'part')} # set of part names
    hide = []
    for inst in ra.instances.values():
        if not hasattr(inst, 'part'):
            continue # skip non-part instances
        if inst.part.name in parts:
            hide.append(inst)
    instance_hide(hide)


def instance_hide_invert():
    from abaqus import session
    vp = session.viewports[session.currentViewportName]
    ra = vp.displayedObject
    displayed = set(vp.assemblyDisplay.visibleInstances) # set of names
    show = []
    for inst in ra.instances.values():
        if not hasattr(inst, 'part'):
            continue # skip non-part instances
        if not inst.name in displayed:
            show.append(inst.name)
    vp.assemblyDisplay.setValues(visibleInstances=show)


def instance_hide_showAll():
    from abaqus import session
    vp = session.viewports[session.currentViewportName]
    ra = vp.displayedObject
    vp.assemblyDisplay.setValues( visibleInstances=list(ra.instances.keys()) )

# {{{1 ASSEMBLY INSTANCES

def instance_reposition_csys(instances, sourceCsys, destinationCsys):
    " Reposition instances based on source and destination datum csys "
    translation = np.asarray(destinationCsys.origin.pointOn) - sourceCsys.origin.pointOn
    R1 = ARotation.from_csys(sourceCsys)
    R2 = ARotation.from_csys(destinationCsys)
    axisDirection, theta = (R2 * (R1.inv())).as_axisAngle()
    for instance in instances:
        instance.ConvertConstraints()
        instance.translate(translation)
        instance.rotateAboutAxis(
                axisPoint=destinationCsys.origin.pointOn,
                axisDirection=axisDirection,
                angle=np.rad2deg(theta),
                )


def instance_matchname():
    " Called by Abaqus/CAE plugin to rename instances based on part names "
    from abaqus import session, mdb, AbaqusException
    vp = session.viewports[session.currentViewportName]
    ra = vp.displayedObject

    parts = {}
    for inst in ra.instances.values():
        if ra.features[inst.name].isSuppressed():
            continue # skip suppressed instances
        if not hasattr(inst, 'part'):
            continue # skip non-part instances
        if not ra.modelName == inst.modelName:
            continue # skip model instances
        parts.setdefault(inst.part.name, []).append(inst.name)

    newNames = []
    for partName, instNames in parts.items():
        for n, instName in enumerate(instNames):
            newName = partName
            if len(instNames) > 1:
                newName += str(-1 - n)
            newNames.append( (instName, newName) )

    print("{} unique parts in {} instances.".format(
        len(parts), len(newNames)))

    instance_rename(newNames=newNames, ra=ra)


def instanceRenameSearch(search, replace=''):
    """Rename according to regular expression sub"""
    import re
    from abaqus import session, mdb, AbaqusException
    vp = session.viewports[session.currentViewportName]
    ra = vp.displayedObject

    baseNames=[]
    unique={}
    for inst in ra.instances.values():
        if ra.features[inst.name].isSuppressed():
            continue # skip suppressed instances
        if not hasattr(inst, 'part'):
            continue # skip non-part instances
        if not ra.modelName == inst.modelName:
            continue # skip model instances
        baseName = re.sub(search, replace, inst.name)
        baseNames.append( (inst.name, baseName) )
        unique.setdefault(baseName, []).append(inst.name)

    newNames=[]
    for instName, baseName in baseNames:
        instList = unique[baseName]
        newName = baseName
        if len(instList) > 1:
            n = instList.index(instName) + 1
            newName += str(-1 * n)
        newNames.append( (instName, newName) )

    newNames = instance_rename(newNames=newNames, ra=ra)


def instance_rename(newNames, ra):
    """Rename instances and correct any references to their regions"""

    from abaqus import session, mdb, AbaqusException
    import inspect

    if False:
        # Search general contact surfaces
        # In 2023 there is no easy way to know if an intance is referenced in general contact
        model.keywordBlock.synchVersions(storeNodesAndElements=False)
        for sie in model.keywordBlock.sieBlocks:
            if sie.startswith('*Contact Initialization Assignment'):
                for n, dataline in enumerate(sie.replace('"', '').split('\n')[1:]):
                    sp = dataline.split(',')
                    # TODO store this data to later check for renamed instance

    def rename(newNames):
        """Rename assembly features
        newNames is a list of (fromName, toName) pairs
        returns list of length newNames containing True and False success
        """
        success = [] # list of success with changing names
        for fromName, toName in newNames:
            if toName is not fromName:
                try:
                    ra.features.changeKey(fromName=fromName, toName=toName)
                    success.append(True)
                    continue
                except (ValueError, AbaqusException) as e:
                    print("Warning: {} {!s}".format(fromName, e))
            success.append(False)
        return success

    # To avoid conflicts, first rename all instances using temporary names
    table = [(old, 'Rename~' + old, new) for old, new in newNames]
    goodList = rename([(old, temp) for old, temp, new in table])
    table = [row for row, good in zip(table, goodList) if good] # only continue with successes

    # Next rename temporary names to the final names
    goodList = rename([(temp, new) for old, temp, new in table])

    # Failures will still have the temporary name; attempt to fix back to original name
    fixTable = [row for row, good in zip(table, goodList) if not good]
    fixedList = rename([(temp, old) for old, temp, new in fixTable]) # go back to original name, if possible
    tempTable = [row for row, fixed in zip(fixTable, fixedList) if not fixed] # instances stuck with temp name

    # Finally, create "renamed" list to include all successful names changes
    table = [row for row, good in zip(table, goodList) if good] # save success
    renamed = [(old, new) for old, temp, new in table]
    renamed.extend([(old, temp) for old, temp, new in tempTable]) # add any failed fixes

    def recursiveSearch(obj):
        """Search each member of obj for regions referring to renamed instances"""
        regionTypes = {1: 'sets', 9: 'surfaces'}
        updates = {}
        nameDict = dict(renamed)
        for attr in dir(obj):
            if attr.startswith('_'):
                continue # skip private members
            try:
                member = getattr(obj, attr)
            except Exception as e:
                continue  # ignore errors
            if callable(member):
                continue  # skip methods
            if not member:
                continue  # skip falsy members
            if hasattr(member, 'getId'):
                continue  # skip symbolic constants
            if isinstance(member, (int, float, str, dict, list)):
                continue  # skip builtin types
            if isinstance(member, tuple):
                if 6 == len(member): # instance regions have length 6
                    setName, partName, instName, regionSpace, regionType, internal = member
                    newInstName = nameDict.get(instName)
                    if not newInstName:
                        continue # this was not a renamed instance
                    if not regionType in regionTypes:
                        print('Warning:', obj.name, attr, 'unknown region type', regionType)
                        continue
                    inst = ra.instances[newInstName]
                    collector = getattr(inst, regionTypes[regionType]) # sets or surfaces
                    if setName in collector.keys():
                        updates[attr] = collector[setName]
                    else:
                        print('Warning:', obj.name, attr, inst.name, 'missing', regionTypes[regionType], setName)
            elif hasattr(member, 'values'):
                for repovalue in member.values():
                    recursiveSearch(repovalue) # search each value in repo
            else:
                recursiveSearch(member)
        if updates and hasattr(obj, 'setValues'):
            try:
                obj.setValues(**updates)
            except (ValueError, AbaqusException) as e:
                print("Warning: {} {!s}".format(obj.name, e))

    if renamed:
        # Search for matching regions in all members of the model
        model = mdb.models[ra.modelName]
        recursiveSearch(model)

    return renamed # list of [fromName, toNamed]


def instance_moveTo(instance, position):
    """Move instance to a specific point in space"""
    from numpy import asarray
    from abaqus import session
    if isinstance(instance, str):
        vp = session.viewports[session.currentViewportName]
        ra = vp.displayedObject # rootAssembly
        instance = ra.instances[instance]
    instance.translate(asarray(position) - instance.getTranslation())


def instance_rotateTo(instance, rotation):
    """Rotate instance to a specific direction in space"""
    import numpy as np
    from abaqus import session
    if isinstance(instance, str):
        vp = session.viewports[session.currentViewportName]
        ra = vp.displayedObject # rootAssembly
        instance = ra.instances[instance]
    rotNew = ARotation.from_rotvec(np.radians(rotation))
    point, direction, deg = instance.getRotation()
    rotOld = ARotation.from_rotvec(np.asarray(direction)*np.radians(deg))
    diff = rotNew*(rotOld.inv()) # calculate difference
    axisDirection, theta = diff.as_axisAngle()
    print(axisDirection, theta)
    instance.rotateAboutAxis(
            axisPoint=instance.getTranslation(),  # rotate about the instance origin
            axisDirection=axisDirection,
            angle=np.rad2deg(theta),
            )


def assembly_derefDuplicate(ra=None, rtol=1e-4, atol=1e-8):
    """Recognize and replace instances of repeated parts with multiple instances of one part.

    Note tighter default rtol and atol since this is automatically checking all instances.
    """

    from abaqus import session, mdb
    if not ra:
        vp = session.viewports[session.currentViewportName]
        ra = vp.displayedObject # rootAssembly
    model = mdb.models[ra.modelName]
    instances = list(ra.instances.values()) # all instances in the assembly
    return instance_derefDup(instances, rtol=rtol, atol=atol)


def instance_derefDup(instances, rtol=1e-2, atol=1e-6):
    """Recognize and replace instances of repeated parts with multiple instances of one part.

    Note looser default rtol and atol since only checking instances that have been specified.
    """

    from abaqus import session, mdb

    vp = session.viewports[session.currentViewportName]
    ra = vp.displayedObject # rootAssembly
    model = mdb.models[ra.modelName]

    # Prepare to prioritize instances according to popularity of their parts
    partInstances = {} # List of all instances associtated to each part
    for inst in ra.instances.values():
        if not hasattr(inst, 'part'):
            continue # skip non-part instances
        partInstances.setdefault(inst.part.name, []).append(inst.name)
    popularity = {}
    for instList in partInstances.values():
        for instName in instList:
            popularity[instName] = len(instList) # Number of instances referring to the same part

    # Loop over specified instances to find similar parts
    partProperties = {}
    vp.disableRefresh()
    vp.disableColorCodeUpdates()
    count = 0
    for inst in statusGenerator(sorted(
            [i for i in instances if hasattr(i, 'part')],
            reverse=True,
            key=lambda i: popularity[i.name] + len(i.nodes) + len(i.surfaces) + len(i.sets)), 'instances checked'):
        if ra.features[inst.name].isSuppressed():
            continue # skip suppressed instances
        if not model.name == inst.modelName:
            continue # skip model instances
        if not inst.dependent:
            continue # skip independent instances
        properties = partProperties.setdefault(inst.part.name, {})
        if not 'mass' in properties:
            # it's a new part
            getMassProperties(inst.part, properties)
            mass = properties.get('mass') or 0
            if mass < atol:
                properties['mass'] = 0
                continue # the part has no mass
            for otherName, otherProps in partProperties.items():
                # It's a new part; check for match with any previous parts
                if inst.part.name == otherName:
                    continue # skip same part
                if 'replacement' in otherProps:
                    continue # skip parts that are replaced by other parts
                if not otherProps.get('mass'):
                    continue # skip parts that don't have mass

                def matching(propertyName, rtol=rtol, atol=atol):
                    "Check for closely matching property values between parts"
                    valueA = properties.get(propertyName)
                    valueB = otherProps.get(propertyName)
                    if type(valueA) is not type(valueB):
                        if DEBUG:
                            print(propertyName, 'different types')
                            print(inst.part.name, valueA)
                            print(otherName, valueB)
                        return False
                    if valueA is None:
                        return True # Both are None
                    if isinstance(valueB, np.ndarray):
                        diff = np.linalg.norm(valueA - valueB)
                        lengthB = np.linalg.norm(valueB)
                        if diff > rtol*lengthB + atol:
                            if DEBUG:
                                print(propertyName, "vectors don't match", rtol, atol)
                                print(inst.part.name, valueA)
                                print(otherName, valueB, diff, rtol*lengthB + atol)
                            return False
                        return True # vector difference is small
                    if not np.allclose(valueB, valueA, rtol=rtol, atol=atol):
                        if DEBUG:
                            print(propertyName, 'does not match', rtol, atol)
                            print(inst.part.name, valueA)
                            print(otherName, valueB, valueB - valueA)
                        return False
                    return True

                if not matching('mass', atol=0):
                    continue # Mass does not a match
                if not matching('principalInertia', atol=0):
                    continue # Different mass properties
                getAreaProperties(inst.part, properties)
                getAreaProperties(model.parts[otherName], otherProps)
                if not matching('area', atol=0):
                    continue # Different area
                getPrincipalDirections(inst.part, properties)
                getPrincipalDirections(model.parts[otherName], otherProps)
                if not matching('orientation', rtol=100*rtol, atol=1e5*atol): # loose tolerance due to compounding error
                    continue # Possible mirror
                properties['replacement'] = otherName
                break # found a match!

        newPartName = properties.get('replacement')
        if not newPartName:
            continue # No replacement found
        newProps = partProperties.get(newPartName)
        newPart = model.parts[newPartName]
        print('Instance {0.name} replaced Part {1.name} with {2.name}'.format(
            inst, inst.part, newPart))
        if DEBUG:
            part_principalProperties(inst.part, properties)
            if not newPart.features.has_key(_principal_csys):
                part_principalProperties(newPart, newProps)

        newCentroid = np.asarray(newProps['volumeCentroid'])
        oldCentroid = np.asarray(properties['volumeCentroid'])

        # Determine location of instance centroid
        inst.ConvertConstraints() # convert any position constraints to absolute positions
        instCentroid = oldCentroid + inst.getTranslation()
        offset, instAxis, instTh = inst.getRotation()
        instRotation = ARotation.from_rotvec(np.asarray(instAxis) * np.radians(instTh))
        instCentroid = instRotation.apply(instCentroid - offset) + offset

        if DEBUG and 'CG-' + inst.part.name not in ra.features:
            pt = ra.ReferencePoint(instCentroid)
            ra.features.changeKey(fromName=pt.name, toName='CG-' + inst.part.name)

        # Replace part and adjust centroid position
        inst.replace(newPart)
        count += 1
        inst.translate(instRotation.apply(oldCentroid - newCentroid))

        # Use principalDirections to correct for rotation difference between parts
        getPrincipalDirections(newPart, newProps)
        newDir = ARotation.from_matrix(newProps['principalDirections'].T)
        getPrincipalDirections(inst.part, properties)
        oldDir = ARotation.from_matrix(properties['principalDirections'].T)
        axis, th = (instRotation * oldDir * newDir.inv() * instRotation.inv()).as_axisAngle()
        inst.rotateAboutAxis(instCentroid, axis, np.degrees(th))

    vp.enableColorCodeUpdates()
    vp.enableRefresh()
    print(count, "instances updated")

# {{{1 ASSEMBLY PARTS

def part_deleteUnused():
    " Called by Abaqus/CAE plugin to remove unused parts "
    from abaqus import session, mdb
    vp = session.viewports[session.currentViewportName]
    ra = vp.displayedObject
    model = mdb.models[ra.modelName]
    parts = set(model.parts.keys())
    used = set()
    for inst in ra.instances.values():
        if hasattr(inst, 'part'):
            used.add(inst.partName)
    unused = parts - used
    vp.disableColorCodeUpdates()
    for partName in statusGenerator(unused, 'unused Parts deleted'):
        del model.parts[partName]
    vp.enableColorCodeUpdates()


def part_edit(instance):
    " Called by Abaqus/CAE plugin to edit part associated with the selcted instance "
    from abaqus import session
    vp = session.viewports[session.currentViewportName]
    ra = vp.displayedObject
    part = instance.part
    count = -1
    for inst in ra.instances.values():
        if hasattr(inst, 'part') and inst.part == part:
            count += 1
    print("instance {} references part {} used by {} other instances.".format(
        instance.name, part.name, count))
    vp.setValues(displayedObject=part)


def part_instanceUnused():
    " Instance parts that are not referenced by any instances "
    from abaqus import session, mdb
    vp = session.viewports[session.currentViewportName]
    ra = vp.displayedObject
    model = mdb.models[ra.modelName]
    unused = set(model.parts.keys()) # set of all part names
    for inst in ra.instances.values():
        if not hasattr(inst, 'part'):
            continue
        if inst.partName in unused:
            unused.remove(inst.partName)
    vp.disableColorCodeUpdates()
    for partName in statusGenerator(sorted(unused), 'instances added'):
        i = 0
        while not i or instName in ra.instances.keys():
            i += 1
            instName = '{}-{}'.format(partName, i)
        ra.Instance(name=instName, part=model.parts[partName], dependent=True)
    vp.enableColorCodeUpdates()


def part_mesh(partNames, refinement=1.0):
    """Generate mesh on listed part names"""
    from abaqus import session, mdb
    from abaqusConstants import SIZE, DEFAULT_SIZE, TET
    vp = session.viewports[session.currentViewportName]
    ra = vp.displayedObject
    model = mdb.models[ra.modelName]
    unmeshed = set()
    for partName in statusGenerator(partNames, 'parts meshed'):
        part = model.parts[partName]
        size = part.getPartSeeds(SIZE) or part.getPartSeeds(DEFAULT_SIZE)
        part.seedPart(refinement*size)
        part.generateMesh()
        if part.getUnmeshedRegions():
            # Try again with TET mesh
            part.setMeshControls(regions=part.cells, elemShape=TET)
            part.generateMesh()
            if part.getUnmeshedRegions():
                unmeshed.add(part.name)
    if unmeshed:
        print(len(unmeshed), 'parts failed to mesh.')
    ra.regenerate()


def part_meshRefine(instances, refinement=0.7):
    """Refine the global mesh size on selected Parts"""
    partNames = {inst.part.name for inst in instances if hasattr(inst, 'part')}
    part_mesh(partNames, refinement)


def part_meshUnmeshed():
    """Generate mesh on unmeshed used Parts and Instances"""
    from abaqus import session
    vp = session.viewports[session.currentViewportName]
    ra = vp.displayedObject
    unmeshedParts = set() # set of used part names
    for inst in ra.instances.values():
        if ra.features[inst.name].isSuppressed():
            continue
        if not hasattr(inst, 'part'):
            continue
        if inst.excludedFromSimulation:
            continue
        if not inst.dependent:
            continue  # TODO mesh independent instances
        if not inst.part.getUnmeshedRegions():
            continue
        unmeshedParts.add(inst.partName)
    if unmeshedParts:
        part_mesh(unmeshedParts)
    else:
        print('No unmeshed dependent parts found')


def part_improveRefinement(instances):
    "Graphically select parts that need better curve refinement"
    from abaqus import session, mdb
    from abaqusConstants import EXTRA_COARSE, COARSE, MEDIUM, FINE, EXTRA_FINE
    refinement = EXTRA_COARSE, COARSE, MEDIUM, FINE, EXTRA_FINE
    vp = session.viewports[session.currentViewportName]
    ra = vp.displayedObject
    model = mdb.models[ra.modelName]
    partNames = {inst.part.name for inst in instances
                 if hasattr(inst, 'part') and inst.part.geometryRefinement is not EXTRA_FINE}
    for partName in statusGenerator(partNames, 'parts refined'):
        part = model.parts[partName]
        index = refinement.index(part.geometryRefinement)
        part.setValues(geometryRefinement=refinement[index + 1])
    if partNames:
        ra.regenerate()
    else:
        print('No parts refined')


def part_resetRefinement():
    """Reset active parts to their default "coarse" level of geometry refinement"""
    from abaqus import session, mdb
    from abaqusConstants import COARSE
    vp = session.viewports[session.currentViewportName]
    ra = vp.displayedObject
    model = mdb.models[ra.modelName]
    partNames = set()
    for inst in ra.instances.values():
        if ra.features[inst.name].isSuppressed():
            continue
        if not hasattr(inst, 'part'):
            continue
        if inst.part.name in partNames:
            continue
        if inst.part.geometryRefinement is not COARSE:
            partNames.add(inst.part.name)
    for partName in statusGenerator(partNames, 'Parts set to coarse geometry refinement'):
        model.parts[partName].setValues(geometryRefinement=COARSE)
    if partNames:
        ra.regenerate()
    else:
        print('No part refinement was reset')

# {{{1 PART

def getAreaProperties(part, properties={}):
    from abaqusConstants import HIGH

    if not properties.get('area'):
        properties.update(
                {k: v for k, v in part.getMassProperties(
                    regions=part.faces,
                    relativeAccuracy=HIGH,
                    specifyDensity=True, density=1).items() if k.startswith('area')
                }
            )
    return properties


def getMassProperties(part, properties={}):
    """Calculate mass properties for given part"""
    from abaqusConstants import HIGH

    if not 'momentOfInertia' in properties:
        properties.update(
            { k: v for k, v in
                part.getMassProperties(
                    relativeAccuracy=HIGH,
                    specifyThickness=True, thickness=0.1,
                    specifyDensity=True, density=1).items()
                if not k.startswith('area')
            }
        )

    if not 'principalInertia' in properties:
        Ixx, Iyy, Izz, Ixy, Iyz, Ixz = properties['momentOfInertia']
        A = np.array([[Ixx, Ixy, Ixz],
                      [Ixy, Iyy, Iyz],
                      [Ixz, Iyz, Izz]])
        if not (None == A).any():
            evalues, evectors = np.linalg.eigh(A)
            # evectors are column eigenvectors such evectors[:,i] corresponds to evalues[i]

            properties['principalInertia'] = evalues
            properties['principalAxes'] = np.ascontiguousarray(evectors.T)

    return properties


def getPrincipalDirections(part, properties={}):
    """Calculate consistent principal directions"""

    if not 'principalDirections' in properties:
        getMassProperties(part, properties)
        getAreaProperties(part, properties)
        o = np.asarray(properties['areaCentroid']) - properties['volumeCentroid'] # for orientation
        if np.allclose(o, np.zeros(3)):
            o = np.ones(3) # use global axis if no centroid difference
        evectors = properties['principalAxes'].T
        d = o.dot(evectors) # project onto principal axes
        evectors *= np.where( d < 0, -1, 1 ) # flip for consistency

        # Ensure right-handed coordinate system
        ax = set( np.argsort( np.abs(d) )[1:] ) # two axes with largest centroid difference
        if not 0 in ax:
            evectors[:,0] = np.cross(evectors[:,1], evectors[:,2])
        elif not 1 in ax:
            evectors[:,1] = np.cross(evectors[:,2], evectors[:,0])
        else:
            evectors[:,2] = np.cross(evectors[:,0], evectors[:,1])
        properties['principalDirections'] = np.ascontiguousarray(evectors.T)
        properties['orientation'] = o.dot(evectors) # project onto principal axes

    return properties


def part_surfaceAreas(part = None):
    " Calculate area of all surfaces in the current part "

    from abaqus import session
    if not part:
        vp = session.viewports[session.currentViewportName]
        part = vp.displayedObject

    print(part.name)
    for surfName in part.surfaces.keys():
        surface = part.surfaces[surfName]
        if len(surface.faces):
            print(surfName, part.getArea(surface.faces))


def part_principalProperties(part = None, properties={}):
    """Calculate and report principal mass properties"""
    from abaqus import session
    from abaqusConstants import CARTESIAN

    if not part:
        vp = session.viewports[session.currentViewportName]
        part = vp.displayedObject
        properties={}

    getPrincipalDirections(part, properties)
    vol = properties.get('volume')
    if not vol:
        raise ZeroDivisionError('Part must have volume')
    mass = properties.get('mass', 0)
    print('{} mass {} (density {})'.format(part.name, mass, mass/vol))
    centroid = np.asarray(properties['volumeCentroid'])
    rot = properties['principalDirections']

    if _principal_csys in part.features:
        del part.features[_principal_csys]
    part.DatumCsysByThreePoints(
            name=_principal_csys,
            coordSysType=CARTESIAN,
            origin=centroid,
            point1=centroid + rot[0], # x direction
            point2=centroid + rot[1], # y direction
        )
    print("\tIx={0[0]}, Iy={0[1]}, Iz={0[2]}".format(properties['principalInertia']))


if __name__ == "__main__":
    import doctest
    DEBUG=False
    doctest.testmod()
