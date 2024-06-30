"""Work-alike subset of scipy.spatial.transform

Carl Osterwisch, April 2022
"""

from __future__ import print_function
import numpy as np

class Rotation:
    """Class to encapsulate basic Quaternion math for 3D rotations

    Carl Osterwisch, June 2020"""


    def __init__(self, q):
        """Quaternion initialization
        
        Note: this checks length but does not normalize its input
        >>> Rotation([1,2,3,4]).as_quat()
        array([1., 2., 3., 4.])
        """
        assert len(q) == 4
        self.q = np.array(q, dtype=np.float64)

    @classmethod
    def identity(cls):
        """ Rotation of zero angle

        >>> R = Rotation.from_rotvec(np.radians([10, 20, 30]))
        >>> I = Rotation.identity()
        >>> np.degrees((R*I).as_rotvec())
        array([10., 20., 30.])
        """
        return cls([1, 0, 0, 0])

    @classmethod
    def from_quat(cls, q):
        """ Initialize normalized rotation using [w, i, j, k]

        >>> Rotation.from_quat([3, 4, 0, 0]).as_quat()
        array([0.6, 0.8, 0. , 0. ])
        """
        return cls(q).normalized()
    
    @classmethod
    def from_rotvec(cls, rotvec):
        """ Initialize using rotation vector

        >>> Rotation.from_rotvec([.1, .2, .3]).as_rotvec()
        array([0.1, 0.2, 0.3])
        >>> R = Rotation.from_rotvec([np.pi/2, 0, 0])
        >>> np.allclose(R.as_matrix(), [[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        True
        """
        vec = np.asarray(rotvec, dtype=np.float64)
        if not len(vec) == 3:
            raise TypeError('rotvec parameter must be length 3')

        angle = np.sqrt(vec.dot(vec)) # vector length
        if 0 == angle:
            return cls.identity()
        axis = vec/angle

        q = np.empty(4, dtype=np.float64)
        q[0] = np.cos(angle/2)
        q[1:] = np.sin(angle/2)*axis
        return cls(q)
  
    def normalized(self):
        """Normalize the rotation quaternion

        >>> Rotation.from_quat([3, 4, 0, 0]).normalized().as_quat()
        array([0.6, 0.8, 0. , 0. ])
        """
        length = np.sqrt(self.q.dot(self.q))
        if length:
            return self.__class__(self.q/length)
        return self.__class__.identity()

    @classmethod
    def from_matrix(cls, d):
        """Method to best-fit quaternion to imperfect rotation matrix.
        Based on https://doi.org/10.2514%2F2.4654

        >>> m = np.array([
        ... [1, 0, 0],
        ... [0, 1, 0],
        ... [0, 0, 1]]).T
        >>> np.allclose(Rotation.from_matrix(m).as_quat(), [1, 0, 0, 0])
        True
        >>> m = np.array([
        ... [0, 1, 0],
        ... [-1, 0, 0],
        ... [0, 0, 1]]).T
        >>> np.allclose(np.degrees(Rotation.from_matrix(m).as_rotvec()), [0, 0, 90])
        True
        """

        d11, d12, d13 = d[0]
        d21, d22, d23 = d[1]
        d31, d32, d33 = d[2]
        k = np.array([
            [d11 + d22 + d33,  d23 - d32,  d31 - d13,  d12 - d21],
            [d23 - d32,  d11 - d22 - d33,  d21 + d12,  d31 + d13],
            [d31 - d13,  d21 + d12,  d22 - d11 - d33,  d32 + d23],
            [d12 - d21,  d31 + d13,  d32 + d23,  d33 - d11 - d22],
        ])/3.0
        assert np.allclose(k, k.T) # check my typing
        evalues, evectors = np.linalg.eigh(k)
        return cls(evectors[:,3]).inv().normalized() # evector corresponding to largest evalue

    def as_matrix(self):
        """Return equivalent rotation matrix
        
        >>> print(Rotation.from_rotvec([0, 0, 0]).as_matrix())
        [[1. 0. 0.]
         [0. 1. 0.]
         [0. 0. 1.]]
        >>> m = Rotation.from_rotvec(np.deg2rad([0, 0, 90])).as_matrix()
        >>> np.allclose(m, np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]).T)
        True
        """
        q0, q1, q2, q3 = self.q
        return np.array([
            [1 - 2*q2*q2 - 2*q3*q3,  2*q1*q2 - 2*q0*q3,  2*q1*q3 + 2*q0*q2],
            [2*q1*q2 + 2*q0*q3,  1 - 2*q1*q1 - 2*q3*q3,  2*q2*q3 - 2*q0*q1],
            [2*q1*q3 - 2*q0*q2,  2*q2*q3 + 2*q0*q1,  1 - 2*q1*q1 - 2*q2*q2],
        ])


    def as_quat(self):
        return np.array(self.q)


    def as_rotvec(self):
        """Convert to rotation vector"""
        v = self.q[1:]
        s = np.sqrt(v.dot(v)) # sin(theta/2)
        if not s:
            return np.array([0., 0., 0.])
        axis = v/s
        theta = 2*np.arctan2(s, self.q[0])
        if abs(theta) > np.pi:
            theta -= np.sign(theta)*2*np.pi
        return theta*axis


    def __repr__(self):
        """Print as rotation vector"""
        return str(self.as_rotvec())


    def __mul__(self, other):
        """Multiply Quaternions to sum their rotations
        
        >>> a = Rotation.from_rotvec(np.radians([0, 0, 30]))
        >>> b = Rotation.from_rotvec(np.radians([0, 90, 0]))
        >>> np.allclose((a*b).as_rotvec(), [-0.41038024,  1.53155991,  0.41038024])
        True
        """
        if not isinstance(other, Rotation):
            raise TypeError('Unsupported multiplication')
        q1 = self.q
        q2 = other.q
        p = np.empty(4)
        p[0] = q1[0]*q2[0] - q1[1:].dot(q2[1:])
        p[1:] = q1[0]*q2[1:] + q2[0]*q1[1:] + np.cross(q1[1:], q2[1:])
        return self.__class__(p)


    def inv(self):
        """Inverse of rotation quaternion
        
        >>> R = Rotation.from_rotvec([0.1, 0.2, 0.3])
        >>> q = (R*R.inv()).as_quat() - Rotation.identity().as_quat()
        >>> np.allclose(q, [0, 0, 0, 0])
        True
        """
        return self.__class__(self.q * [1, -1, -1, -1])


    def apply(self, pt):
        """Apply rotation to a 3D point
        
        Example: Define rotation R about axis x=y
        >>> R = Rotation.from_rotvec(np.array([1, 1, 0])*np.pi/np.sqrt(2))
        >>> np.allclose(R.apply([10, 0, 0]), [0, 10, 0])
        True
        >>> np.allclose(R.apply([0, 0, 100]), [0, 0, -100])
        True
        """

        i, j, k = pt
        v = self.__class__([0.0, i, j, k])
        return (self.inv() * v * self).q[1:]


if __name__ == "__main__":
    import doctest
    doctest.testmod()
