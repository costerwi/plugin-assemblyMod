"""Script to adjust assembly structure.

Carl Osterwisch <costerwi@gmail.com>
"""

def instance_delete(instances):
    " Called by Abaqus/Viewer plugin to remove selected instances "
    from abaqus import session
    #import abaqusConstants

    print len(instances), "instances selected for deletion."

    vp = session.viewports[session.currentViewportName]
    ra = vp.displayedObject
    vp.disableRefresh()
    vp.disableColorCodeUpdates()
    for i in instances:
        del ra.features[i.name]
    vp.enableColorCodeUpdates()
    vp.enableRefresh()

def part_deleteUnused():
    " Called by Abaqus/Viewer plugin to remove unused parts "
    from abaqus import session, mdb
    vp = session.viewports[session.currentViewportName]
    ra = vp.displayedObject
    model = mdb.models[ra.modelName]
    parts = set(model.parts.keys())
    used = set([inst.partName for inst in ra.instances.values()])
    unused = parts - used
    print len(unused), "unused parts to be deleted."
    for partName in unused:
        del model.parts[partName]

