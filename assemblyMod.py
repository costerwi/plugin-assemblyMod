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
                    specifyDensity=True, density=1).items()
                if not k.startswith('area')
            }
        )

    if not 'principalInertia' in properties:
        Ixx, Iyy, Izz, Ixy, Iyz, Ixz = properties['momentOfInertia']
        A = np.array([[Ixx, Ixy, Ixz],
                      [Ixy, Iyy, Iyz],
                      [Ixz, Iyz, Izz]])
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
        ax = set( np.argsort( np.abs(d) )[:2] ) # two axes with largest centroid difference
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


def part_derefDuplicate(ra=None, rtol=1e-6, atol=1e-8):
    " Recognize and replace instances of repeated parts with multiple instances of one part "
    from time import time
    from abaqus import session, mdb
    from abaqusConstants import HIGH

    if not ra:
        vp = session.viewports[session.currentViewportName]
        ra = vp.displayedObject # rootAssembly
    model = mdb.models[ra.modelName]

    partProperties = {}
    vp.disableRefresh()
    vp.disableColorCodeUpdates()
    count = 0
    t0 = time()
    for inst in ra.instances.values():
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
            mass = properties.get('mass', 0)
            if mass < 10*atol:
                continue # the part has no mass
            for otherName, otherProps in partProperties.items():
                # It's a new part; check for match with any previous parts
                if inst.part.name == otherName:
                    continue # skip same part
                if 'replacement' in otherProps:
                    continue # skip parts that are replaced by other parts
                if  not np.allclose(mass, otherProps['mass'], rtol=rtol, atol=atol):
                    continue # Mass does not a match
                inertia = properties.get('principalInertia', 0)
                otherInertia = otherProps.get('principalInertia', 0)
                if not np.allclose(inertia, otherInertia, rtol=rtol, atol=atol):
                    continue # Different mass properties
                area = getAreaProperties(inst.part, properties).get('area', 0)
                otherArea = getAreaProperties(model.parts[otherName], otherProps).get('area', 0)
                if not np.allclose(area, otherArea, rtol=rtol, atol=atol): # Surface area doesn't match
                    continue
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
        instRotation = ARotation.from_rotvec(np.asarray(instAxis) * np.radians(instTh)).inv()
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
        axis, th = (instRotation.inv() * (newDir * oldDir) * instRotation).as_axisAngle()
        inst.rotateAboutAxis(instCentroid, axis, np.degrees(th))

    vp.enableColorCodeUpdates()
    vp.enableRefresh()
    print("{} instances updated in {:.1f} seconds".format(count, time() - t0))
