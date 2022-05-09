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
    def from_matrix(cls, matrix):
        """Initialize from rotation matrix

        >>> m = ARotation.from_rotvec([0.2, 0.3, 0.4]).as_matrix()
        >>> R = ARotation.from_matrix(m)
        >>> R.__class__
        <class '__main__.ARotation'>
        >>> R.as_rotvec()
        array([0.2, 0.3, 0.4])
        """

        if hasattr(Rotation, 'from_matrix'):
            return cls.from_quat(Rotation.from_matrix(matrix).as_quat())
        if hasattr(cls, 'align_vectors'):
            return cls.align_vectors(matrix.T, np.eye(3))[0] #workaround 1
        return cls.match_vectors(matrix.T, np.eye(3))[0] #workaround 2

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
    print("{}/{} instances selected for suppression.".format(
        len(instances), len(ra.instances)))
    ra.suppressFeatures([inst.name for inst in instances])


def instance_suppress_part(instances):
    " Called by Abaqus/CAE plugin to suppress selected instances "
    from abaqus import session
    vp = session.viewports[session.currentViewportName]
    ra = vp.displayedObject
    parts = set([inst.part.name for inst in instances]) # set of part names
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
    print("{}/{} instances hidden.".format(
        len(instances), len(ra.instances)))
    vp.assemblyDisplay.hideInstances([inst.name for inst in instances])


def instance_hide_part(instances):
    " Called by Abaqus/CAE plugin to hide parts"
    from abaqus import session
    vp = session.viewports[session.currentViewportName]
    ra = vp.displayedObject
    parts = set([inst.part.name for inst in instances]) # set of part names
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
    vp.assemblyDisplay.setValues( visibleInstances=ra.instances.keys() )

# {{{1 ASSEMBLY INSTANCES

def instance_editPart(instance):
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


def instance_reposition(instances, sourceCsys, destinationCsys):
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
    from math import log10
    from abaqus import session
    vp = session.viewports[session.currentViewportName]
    ra = vp.displayedObject

    parts = {}  # Identify unique parts and their instances
    for n, inst in enumerate(ra.instances.values()):
        if ra.features[inst.name].isSuppressed():
            continue
        if not hasattr(inst, 'part'):
            continue # skip non-part instances
        tempName = 'temp~{}'.format(n)
        parts.setdefault(inst.partName, []).append(tempName)
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


def assembly_derefDuplicate(ra=None, rtol=1e-6, atol=1e-8):
    """Recognize and replace instances of repeated parts with multiple instances of one part.

    Note tighter default rtol and atol since this is automatically checking all instances.
    """

    from abaqus import session, mdb
    if not ra:
        vp = session.viewports[session.currentViewportName]
        ra = vp.displayedObject # rootAssembly
    model = mdb.models[ra.modelName]
    instances = ra.instances.values() # all instances in the assembly
    return instance_commonPart(instances, rtol=rtol, atol=atol)


def instance_derefDup(instances, rtol=1e-2, atol=1e-8):
    """Recognize and replace instances of repeated parts with multiple instances of one part.

    Note looser default rtol and atol since only checking instances that have been specified.
    """

    from time import time
    from abaqus import session, mdb
    from abaqusConstants import HIGH

    vp = session.viewports[session.currentViewportName]
    ra = vp.displayedObject # rootAssembly
    model = mdb.models[ra.modelName]
    partProperties = {}
    vp.disableRefresh()
    vp.disableColorCodeUpdates()
    count = 0
    t0 = time()
    for inst in instances:
        if ra.features[inst.name].isSuppressed():
            continue # skip suppressed instances
        if not hasattr(inst, 'part'):
            continue # skip non-part instances
        if not inst.dependent:
            continue # skip independent instances
        properties = partProperties.setdefault(inst.part.name, {})
        if not 'mass' in properties:
            # it's a new part
            getMassProperties(inst.part, properties)
            mass = properties.get('mass') or 0
            if mass < 10*atol:
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
                if  not np.allclose(mass, otherProps['mass'], rtol=rtol, atol=atol):
                    continue # Mass does not a match
                inertia = properties.get('principalInertia')
                otherInertia = otherProps.get('principalInertia')
                if np.any(None == inertia) or np.any(None == otherInertia):
                    continue # Must both have inertia
                if not np.allclose(inertia, otherInertia, rtol=rtol, atol=atol):
                    continue # Different mass properties
                area = getAreaProperties(inst.part, properties).get('area')
                otherArea = getAreaProperties(model.parts[otherName], otherProps).get('area')
                if not (area and otherArea):
                    continue # Must both have area
                if not np.allclose(area, otherArea, rtol=rtol, atol=atol): # Surface area doesn't match
                    continue # Different area
                properties['replacement'] = otherName
                break # found a match!

        newPartName = properties.get('replacement')
        if not newPartName:
            continue # No replacement found
        newProps = partProperties.get(newPartName)
        newPart = model.parts[newPartName]
        print('Instance {0.name} replaced {1.name} with {2.name}'.format(
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

        if DEBUG and not ra.features.has_key('CG-' + inst.part.name):
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
    print("{} instances updated in {:.1f} seconds".format(count, time() - t0))

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
    print("{}/{} parts deleted.".format(
        len(unused), len(parts)))
    vp.disableColorCodeUpdates()
    for partName in unused:
        del model.parts[partName]
    vp.enableColorCodeUpdates()


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
    for partName in sorted(unused):
        i = 0
        while not i or instName in ra.instances.keys():
            i += 1
            instName = '{}-{}'.format(partName, i)
        ra.Instance(name=instName, part=model.parts[partName], dependent=True)
    print('Added {} instances'.format(len(unused)))

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

    if part.features.has_key(_principal_csys):
        del part.features[_principal_csys]
    part.DatumCsysByThreePoints(
            name=_principal_csys,
            coordSysType=CARTESIAN,
            origin=centroid,
            point1=centroid + rot[0], # x direction
            point2=centroid + rot[1], # y direction
        )
    print("\tIx={0[0]}, Iy={0[1]}, Iz={0[2]}".format(properties['principalInertia']))
