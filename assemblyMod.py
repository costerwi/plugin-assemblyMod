"""Script to adjust Abaqus/CAE assembly structure.

Carl Osterwisch <costerwi@gmail.com> December 2013
"""

from __future__ import print_function
import os
import numpy as np
DEBUG = os.environ.get('DEBUG')

class Rotation:
    """Class to encapsulate basic Quaternion math for 3D rotations

    Carl Osterwisch, June 2020"""

    def __init__(self, vec):
        """Quaternion initialization"""
        vec = np.asarray(vec)
        if 4 == len(vec): # Initialize with Quaternion vector
            self.q = vec
        elif 3 == len(vec): # Initialize with rotation vector
            theta = np.sqrt(vec.dot(vec))
            axis = vec/(theta or 1)
            self.q = np.empty(4)
            self.q[0] = np.cos(theta/2)
            self.q[1:] = np.sin(theta/2)*axis
        else:
            raise TypeError('vec parameter must be length 3 (rotation vector) or 4 (quaternion)')

    @staticmethod
    def fromRotMatrix(m):
        m00, m01, m02 = m[0]
        m10, m11, m12 = m[1]
        m20, m21, m22 = m[2]
        p = np.empty(4)
        p[0] = np.sqrt(1 + m00 + m11 + m22)*0.5
        p[1:] = np.array([m21 - m12, m02 - m20, m10 - m01])/(4*p[0])
        return Rotation(p)

    @staticmethod
    def fromCsys(csys):
        m = np.array([
            csys.axis1.direction,
            csys.axis2.direction,
            csys.axis3.direction,
        ]).transpose()
        return Rotation.fromRotMatrix(m)

    def axisAngle(self):
        """Convert to rotation around axis, theta in radians"""
        v = self.q[1:]
        s = np.sqrt(v.dot(v)) # sin(theta/2)
        if not s:
            v += (0, 0, 1) # Unit length
        c = self.q[0] # cos(theta/2)
        theta = 2*np.arctan2(s, c)
        axis = v/(s or 1)
        return axis, theta

    def vec(self):
        """Convert to rotation vector"""
        axis, theta = self.axisAngle()
        return theta*axis

    def __repr__(self):
        """Print as rotation vector"""
        return str(self.vec())

    def __mul__(self, other):
        """Scalar multiplication of rotation angle"""
        return Rotation(other*self.vec())

    def __rmul__(self, other):
        """Scalar multiplication of rotation angle"""
        return self*other

    def __add__(self, other):
        """Multiply Quaternions to sum their rotations"""
        if not isinstance(other, Rotation):
            raise TypeError('Unsupported Rotation addition')
        # Note: Order of operands is flipped here since class is oriented toward
        # rotation vectors and left to right order is more natural.
        q1 = other.q # new rotation
        q2 = self.q # old rotation
        p = np.empty(4)
        p[0] = q1[0]*q2[0] - q1[1:].dot(q2[1:])
        p[1:] = q1[0]*q2[1:] + q2[0]*q1[1:] + np.cross(q1[1:], q2[1:])
        return Rotation(p)

    def inv(self):
        """Inverse"""
        return Rotation(self.q * [1, -1, -1, -1])

    def __sub__(self, other):
        """Difference between rotation vectors"""
        if not isinstance(other, Rotation):
            raise TypeError('Unsupported Rotation difference')
        return other.inv() + self

    def apply(self, pt):
        """Apply rotation to a point"""
        v = Rotation([0, pt[0], pt[1], pt[2]])
        return (self.inv() + v + self).q[1:]


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


def instance_reposition(instances, sourceCsys, destinationCsys):
    " Reposition instances based on source and destination datum csys "
    translation = np.asarray(destinationCsys.origin.pointOn) - sourceCsys.origin.pointOn
    axisDirection, theta = (Rotation.fromCsys(destinationCsys) - Rotation.fromCsys(sourceCsys)).axisAngle()
    for instance in instances:
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


def part_principalProperties():
    """Calculate and report principal mass properties"""
    from abaqus import session
    from abaqusConstants import CARTESIAN

    vp = session.viewports[session.currentViewportName]
    part = vp.displayedObject

    massProp = part.queryGeometry(printResults=False)
    vol = massProp.get('volume')
    if not vol:
        raise ZeroDivisionError('Part must have volume')
    massProp.update( getMassProperties(part) )
    mass = massProp.get('mass', 0)
    print('{} mass {} (density {})'.format(part.name, mass, mass/vol))
    centroid = np.asarray(massProp['centerOfMass'])
    rot = massProp['principalDirections']

    v = np.mean(massProp['boundingBox'], axis=0) - massProp['volumeCentroid']
    rot = massProp['principalDirections']
    if v.dot(rot[1]) < 0:
        # Flip Y
        rot[1] *= -1
        if DEBUG:
            print('Flipped', rot[0])
    rot[2] = np.cross(rot[0], rot[1])

    name = 'Principal csys'
    if part.features.has_key(name):
        del part.features[name]
    part.DatumCsysByThreePoints(
            name=name,
            coordSysType=CARTESIAN,
            origin=centroid,
            point1=centroid + rot[0], # x direction
            point2=centroid + rot[1], # y direction
        )
    print("\tIx={0[0]}, Iy={0[1]}, Iz={0[2]}".format(massProp['principalInertia']))


def getMassProperties(part):
    """Calculate mass properties for given part"""
    from abaqusConstants import HIGH

    massProp = part.getMassProperties(
            relativeAccuracy=HIGH,
            specifyDensity=True, density=1)
    Ixx, Iyy, Izz, Ixy, Iyz, Ixz = massProp['momentOfInertia']

    A = np.array([[Ixx, Ixy, Ixz],
                  [Ixy, Iyy, Iyz],
                  [Ixz, Iyz, Izz]])
    evalues, evectors = np.linalg.eigh(A)
    # evectors are column eigenvectors such evectors[:,i] corresponds to evalues[i]

    massProp['principalInertia'] = evalues
    massProp['principalDirections'] = np.ascontiguousarray(evectors.transpose())
    return massProp


def part_surfaceAreas():
    " Calculate area of all surfaces in the current part "
    from abaqus import session
    vp = session.viewports[session.currentViewportName]
    part = vp.displayedObject

    print(part.name)
    for surfName in part.surfaces.keys():
        surface = part.surfaces[surfName]
        if len(surface.faces):
            print(surfName, part.getArea(surface.faces))


def part_derefDuplicate():
    " Replace instances of repeated parts with multiple instances of one part "
    from abaqus import session
    from abaqusConstants import HIGH

    vp = session.viewports[session.currentViewportName]
    ra = vp.displayedObject

    partNames = set()
    volumeParts = []
    volumeInstances = []
    for inst in ra.instances.values():
        if ra.features[inst.name].isSuppressed():
            continue
        try:
            part = inst.part
        except AttributeError:
            continue
        volumeInstances.append(inst)
        if part.name in partNames:
            continue # Already calculated this part
        partNames.add(part.name)
        properties = part.queryGeometry(printResults=False)
        if properties.get('volume'):
            # List solid parts
            volumeParts.append( (part, properties) )

    vp.disableRefresh()
    vp.disableColorCodeUpdates()
    count = 0
    while len(volumeParts) > 1:
        masterPart, masterProp = volumeParts.pop(0)
        unmatched = [] # Keep group of parts which do not match the current master
        for slavePart, slaveProp in volumeParts:
            unmatched.append( (slavePart, slaveProp) )
            if abs(masterProp['volume'] - slaveProp['volume'])/masterProp['volume'] > 0.002: # volumes don't match
                continue # Not a match

            # Calculate and compare surface areas
            for part, properties in ( (masterPart, masterProp), (slavePart, slaveProp) ):
                if not properties.get('area'):
                    properties['area'] = part.getMassProperties(
                        regions=part.faces,
                        relativeAccuracy=HIGH).get('area')
            if abs(masterProp['area'] - slaveProp['area'])/masterProp['area'] > 0.002: # Surface area doesn't match
                continue # Not a match

            # Calculate and compare principal moments of inertia
            for part, properties in ( (masterPart, masterProp), (slavePart, slaveProp) ):
                if not 'principalInertia' in properties:
                    try:
                        properties.update(getMassProperties(part))
                    except:
                        unmatched.pop() # Forget this part
                        continue
                    v = np.mean(properties['boundingBox'], axis=0) - properties['volumeCentroid']
                    evectors = properties['principalDirections']
                    if v.dot(evectors[1]) < 0:
                        # Flip Y
                        evectors[1] *= -1
                    evectors[2] = np.cross(evectors[0], evectors[1])
            if not np.allclose(slaveProp['principalInertia'], masterProp['principalInertia'], rtol=1e-5):
                continue # Not a match

            unmatched.pop() # It's a match!

            # Replace all Instances of this slavePart with the masterPart.
            # The difference in center of mass will be used to position the masterPart.
            masterCentroid = np.asarray(masterProp['centerOfMass'])
            slaveCentroid = np.asarray(slaveProp['centerOfMass'])
            for inst in [i for i in volumeInstances if i.part == slavePart]:
                print('Instance {0.name} replaced {1.name} with {2.name}'.format(
                    inst, slavePart, masterPart))

                # Determine location of instance centroid
                instCentroid = slaveCentroid + inst.getTranslation()
                offset, instAxis, instTh = inst.getRotation()
                instRotation = Rotation(np.asarray(instAxis) * np.deg2rad(instTh))
                instCentroid = instRotation.apply(instCentroid - offset) + offset

                if DEBUG:
                    pt = ra.ReferencePoint(instCentroid)
                    ra.features.changeKey(fromName=pt.name, toName='CG-' + slavePart.name)

                # Replace part and adjust position
                inst.replace(masterPart)
                inst.translate(instRotation.apply(slaveCentroid - masterCentroid))
                count += 1

                # Use principalDirections to correct for rotation difference between parts
                mdir = Rotation.fromRotMatrix(masterProp['principalDirections'].transpose())
                sdir = Rotation.fromRotMatrix(slaveProp['principalDirections'].transpose())
                axis, th = (instRotation - (mdir - sdir) - instRotation).axisAngle()
                inst.rotateAboutAxis(instCentroid, axis, np.rad2deg(th))

        volumeParts = unmatched # Continue to process any remaining parts

    vp.enableColorCodeUpdates()
    vp.enableRefresh()
    print("{} instances updated.".format(count))
