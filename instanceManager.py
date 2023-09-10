
import numpy as np
import customKernel

def init():
    updateData()

def updateData():
    from abaqus import session
    viewport = session.viewports[session.currentViewportName]
    rootAssembly = viewport.displayedObject
    instanceData = []
    for i in rootAssembly.instances.values():
        Tx, Ty, Tz = i.getTranslation()
        a, b, deg = i.getRotation()
        Rx, Ry, Rz = np.asarray(b)*deg
        instanceData.append( (i.name, Tx, Ty, Tz, Rx, Ry, Rz ) )
    session.customData.instanceData = instanceData

oldInstanceName = ''
def outline(instanceName):
    # TODO highlight mutiple selected instances
    from abaqus import session, highlight, unhighlight
    global oldInstanceName
    vp = session.viewports[session.currentViewportName]
    ra = vp.displayedObject # rootAssembly
    if oldInstanceName in ra.instances.keys():
        unhighlight(ra.instances[oldInstanceName])
    else:
        for instance in ra.instances.values():
            unhighlight(instance)  # unhighlight all instances
    instance = ra.instances[instanceName]
    highlight(instance)  # highlight this one instance
    oldInstanceName = instanceName
