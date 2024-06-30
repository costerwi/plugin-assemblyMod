"""Define the AFXProcedure class to handle assembly modifications.

Carl Osterwisch <costerwi@gmail.com> November 2013
"""
__version__ = "0.10.0"
helpUrl='https://github.com/costerwi/plugin-assemblyMod'

from abaqusGui import *

###########################################################################
# {{{1 Procedure definition
###########################################################################
class InstanceSelectProcedure(AFXProcedure):
    """Base class to allow user to select Instances and run an assemblyMod command"""

    prompt = 'instances to operate upon' # must be defined by child class
    method = 'thing_to_do' # must be defined by child class

    def __init__(self, owner):
        AFXProcedure.__init__(self, owner) # Construct the base class.

        self.command = AFXGuiCommand(mode=self,
                method=self.method,
                objectName='assemblyMod',
                registerQuery=FALSE)

        objectToPick = self.prompt.split()[0]
        if objectToPick.endswith('s'): # plural
            self.numberToPick = MANY
        else:
            self.numberToPick = ONE
        self.instancesKw = AFXObjectKeyword(
                command=self.command,
                name=objectToPick,
                isRequired=TRUE)

    def getFirstStep(self):
        self.step1 = AFXPickStep(
                owner=self,
                keyword=self.instancesKw,
                prompt='Select ' + self.prompt,
                entitiesToPick=INSTANCES,
                numberToPick=self.numberToPick,
                sequenceStyle=TUPLE)    # TUPLE or ARRAY
        return self.step1

    def getLoopStep(self):
        if MANY == self.numberToPick:
            return self.step1  # loop until canceled


class instanceRepositionCsysProcedure(AFXProcedure):
    "Class to start the instance selection procedure"

    def __init__(self, owner):
        AFXProcedure.__init__(self, owner) # Construct the base class.

        # Command
        instance_cmd = AFXGuiCommand(mode=self,
                method='instance_reposition_csys',
                objectName='assemblyMod',
                registerQuery=FALSE)

        # Keywords
        self.instancesKw = AFXObjectKeyword(
                command=instance_cmd,
                name='instances',
                isRequired=TRUE)

        self.sourceCsysKw = AFXObjectKeyword(
                command=instance_cmd,
                name='sourceCsys',
                isRequired=TRUE)

        self.destinationCsysKw = AFXObjectKeyword(
                command=instance_cmd,
                name='destinationCsys',
                isRequired=TRUE)

    def getFirstStep(self):
        self.step1 = AFXPickStep(
                owner=self,
                keyword=self.instancesKw,
                prompt="Select instances to reposition",
                entitiesToPick=INSTANCES,
                numberToPick=MANY,
                sequenceStyle=TUPLE)    # TUPLE or ARRAY
        return self.step1

    def getNextStep(self, previousStep):
        if previousStep == self.step1:
            self.step2 = AFXPickStep(
                    owner=self,
                    keyword=self.sourceCsysKw,
                    prompt="Select source csys",
                    entitiesToPick=DATUM_CSYS,
                    )
            return self.step2
        elif previousStep == self.step2:
            return AFXPickStep(
                    owner=self,
                    keyword=self.destinationCsysKw,
                    prompt="Select destination csys",
                    entitiesToPick=DATUM_CSYS,
                    )
        return None # no more steps


class InstanceRenameProcedure(AFXProcedure):
    class Dialog1(AFXDataDialog):
        def __init__(self, procedure):
            AFXDataDialog.__init__(
                self, procedure, 'Rename Instances',
                self.OK|self.CANCEL, DIALOG_NORMAL|DECOR_RESIZE,
                )
            p = AFXVerticalAligner(self, opts=LAYOUT_FILL_X)
            AFXTextField(p, 20, 'Search',
                procedure.searchKw,
                opts=LAYOUT_FILL_X)
            AFXTextField(p, 20, 'Replace',
                procedure.replaceKw,
                opts=LAYOUT_FILL_X)

    def __init__(self, owner):
        AFXProcedure.__init__(self, owner) # Construct the base class.
        command = AFXGuiCommand(mode=self,
                method='instanceRenameSearch',
                objectName='assemblyMod',
                registerQuery=FALSE)
        self.searchKw = AFXStringKeyword(
                command=command,
                name='search',
                isRequired=TRUE)
        self.replaceKw = AFXStringKeyword(
                command=command,
                name='replace',
                isRequired=TRUE)

    def getFirstStep(self):
        return AFXDialogStep(
            owner=self,
            dialog=self.Dialog1(self),
            prompt='Enter search and replace text strings or regular expressions',
            )


###########################################################################
# Register the plugins
###########################################################################
toolset = getAFXApp().getAFXMainWindow().getPluginToolset()

# {{{1 ASSEMBLY INSTANCES

menu = ['&Instances']

toolset.registerKernelMenuButton(
        buttonText='|'.join(menu) + '|Find &duplicate Parts',
        moduleName='assemblyMod',
        functionName='assembly_derefDuplicate()',
        author='Carl Osterwisch',
        version=__version__,
        helpUrl=helpUrl,
        applicableModules=['Assembly'],
        description='Use mass properties to automatically identify and replace instances of similar parts with multiple instances of one part. '
            'Two parts must have the same mass, area, and primary moments of inertia within 0.01% to be recognized as equal.'
        )

class InstanceDuplicatePicked(InstanceSelectProcedure):
        prompt = 'instances to search for common parts'
        method = 'instance_derefDup'

toolset.registerGuiMenuButton(
        buttonText='|'.join(menu) + '|&Pick Duplicate Parts...',
        object=InstanceDuplicatePicked(toolset),
        kernelInitString='import assemblyMod',
        author='Carl Osterwisch',
        version=__version__,
        helpUrl=helpUrl,
        applicableModules=['Assembly'],
        description='Graphically select instances to replace with a common part. '
            'Replacement will not happen if mass properties are significantly different from each other. '
            'Two parts must have the same mass, area, and primary moments of inertia within 1% to be recognized as equal.'
        )

toolset.registerGuiMenuButton(
        buttonText='|'.join(menu) + '|&Reposition using 2 csys...',
        object=instanceRepositionCsysProcedure(toolset),
        kernelInitString='import assemblyMod',
        author='Carl Osterwisch',
        version=__version__,
        helpUrl=helpUrl,
        applicableModules=['Assembly'],
        description='Reposition instances based on position and orientation of selected source and destination csys.'
        )

toolset.registerKernelMenuButton(
        buttonText='|'.join(menu) + '|Rename using Part &name',
        moduleName='assemblyMod',
        functionName='instance_matchname()',
        author='Carl Osterwisch',
        version=__version__,
        helpUrl=helpUrl,
        applicableModules=['Assembly'],
        description='Update instance names using part name as a base.')

toolset.registerGuiMenuButton(
        buttonText='|'.join(menu) + '|&Rename using search/replace...',
        object=InstanceRenameProcedure(toolset),
        kernelInitString='import assemblyMod',
        author='Carl Osterwisch',
        version=__version__,
        helpUrl=helpUrl,
        applicableModules=['Assembly'],
        description='Rename instances using specified criteria'
        )

# {{{1 ASSEMBLY INSTANCES DELETE

menu.append('&Delete')

toolset.registerKernelMenuButton(
        buttonText='|'.join(menu) + '|&Hidden',
        moduleName='assemblyMod',
        functionName='instance_delete_hidden()',
        author='Carl Osterwisch',
        version=__version__,
        helpUrl=helpUrl,
        applicableModules=['Assembly'],
        description='Delete instances that are currently hidden.')


toolset.registerKernelMenuButton(
        buttonText='|'.join(menu) + '|&Suppressed',
        moduleName='assemblyMod',
        functionName='instance_delete_suppressed()',
        author='Carl Osterwisch',
        version=__version__,
        helpUrl=helpUrl,
        applicableModules=['Assembly'],
        description='Delete instances that are currently suppressed.')

menu.pop()

# {{{1 ASSEMBLY INSTANCES SUPPRESS

menu.append('&Suppress')

class InstanceSuppressPicked(InstanceSelectProcedure):
        prompt = 'instances to suppress'
        method = 'instance_suppress'

toolset.registerGuiMenuButton(
        buttonText='|'.join(menu) + '|&Picked Instances...',
        object=InstanceSuppressPicked(toolset),
        kernelInitString='import assemblyMod',
        author='Carl Osterwisch',
        version=__version__,
        helpUrl=helpUrl,
        applicableModules=['Assembly', 'Interaction', 'Load', 'Mesh'],
        description='Graphically select instances to suppress in the assembly.'
        )

class InstanceSuppressPart(InstanceSelectProcedure):
        prompt = 'instances of parts to suppress'
        method = 'instance_suppress_part'

toolset.registerGuiMenuButton(
        buttonText='|'.join(menu) + '|All Instances of &Part...',
        object=InstanceSuppressPart(toolset),
        kernelInitString='import assemblyMod',
        author='Carl Osterwisch',
        version=__version__,
        helpUrl=helpUrl,
        applicableModules=['Assembly', 'Interaction', 'Load', 'Mesh'],
        description='Select parts to suppress their instancse.'
        )

toolset.registerKernelMenuButton(
        buttonText='|'.join(menu) + '|Zero &Area Instances',
        moduleName='assemblyMod',
        functionName='instance_suppress_noArea()',
        author='Carl Osterwisch',
        version=__version__,
        helpUrl=helpUrl,
        applicableModules=['Assembly', 'Interaction', 'Load'],
        description='Suppress instances that have no surface area.')

toolset.registerKernelMenuButton(
        buttonText='|'.join(menu) + '|Zero &Volume Instances',
        moduleName='assemblyMod',
        functionName='instance_suppress_noVolume()',
        author='Carl Osterwisch',
        version=__version__,
        helpUrl=helpUrl,
        applicableModules=['Assembly', 'Interaction', 'Load', 'Mesh'],
        description='Suppress instances that have no volume.')

toolset.registerKernelMenuButton(
        buttonText='|'.join(menu) + '|&Unmeshed parts',
        moduleName='assemblyMod',
        functionName='instance_suppress_unmeshed()',
        author='Carl Osterwisch',
        version=__version__,
        helpUrl=helpUrl,
        applicableModules=['Assembly', 'Interaction', 'Load', 'Mesh'],
        description='Suppress instances of parts which have unmeshed regions.')

toolset.registerKernelMenuButton(
        buttonText='|'.join(menu) + '|Containing elements of failed &quality',
        moduleName='assemblyMod',
        functionName='instance_suppress_badElements()',
        author='Carl Osterwisch',
        version=__version__,
        helpUrl=helpUrl,
        applicableModules=['Assembly', 'Interaction', 'Load', 'Mesh'],
        description='Suppress instances of parts which have unmeshed regions.')

toolset.registerKernelMenuButton(
        buttonText='|'.join(menu) + '|&Invert',
        moduleName='assemblyMod',
        functionName='instance_suppress_invert()',
        author='Carl Osterwisch',
        version=__version__,
        helpUrl=helpUrl,
        applicableModules=['Assembly', 'Interaction', 'Load', 'Mesh'],
        description='Resume suppressed instances and suppress active instances.')

toolset.registerKernelMenuButton(
        buttonText='|'.join(menu) + '|&Resume all',
        moduleName='assemblyMod',
        functionName='instance_suppress_resumeAll()',
        author='Carl Osterwisch',
        version=__version__,
        helpUrl=helpUrl,
        applicableModules=['Assembly', 'Interaction', 'Load', 'Mesh'],
        description='Resume suppressed instances.')

menu.pop()

# {{{1 ASSEMBLY INSTANCES HIDE

menu.append('&Hide')

class InstanceHidePicked(InstanceSelectProcedure):
        prompt = 'instances to hide'
        method = 'instance_hide'

toolset.registerGuiMenuButton(
        buttonText='|'.join(menu) + '|&Picked Instances...',
        object=InstanceHidePicked(toolset),
        kernelInitString='import assemblyMod',
        author='Carl Osterwisch',
        version=__version__,
        helpUrl=helpUrl,
        applicableModules=['Assembly', 'Interaction', 'Load'],
        description='Graphically select instances to hide from the assembly.' \
                ' See also: right-click menu, Hide Instance.'
        )

class InstanceHidePart(InstanceSelectProcedure):
        prompt = 'instances of parts to hide'
        method = 'instance_hide_part'

toolset.registerGuiMenuButton(
        buttonText='|'.join(menu) + '|All Instances of &Part...',
        object=InstanceHidePart(toolset),
        kernelInitString='import assemblyMod',
        author='Carl Osterwisch',
        version=__version__,
        helpUrl=helpUrl,
        applicableModules=['Assembly', 'Interaction', 'Load'],
        description='Hide visible instances and show hidden instances.'
        )

toolset.registerKernelMenuButton(
        buttonText='|'.join(menu) + '|&Invert',
        moduleName='assemblyMod',
        functionName='instance_hide_invert()',
        author='Carl Osterwisch',
        version=__version__,
        helpUrl=helpUrl,
        applicableModules=['Assembly', 'Interaction', 'Load'],
        description='Show hidden instances and hide shown instances.')

toolset.registerKernelMenuButton(
        buttonText='|'.join(menu) + '|&Show all',
        moduleName='assemblyMod',
        functionName='instance_hide_showAll()',
        author='Carl Osterwisch',
        version=__version__,
        helpUrl=helpUrl,
        applicableModules=['Assembly', 'Interaction', 'Load'],
        description='Show all hidden instances.')

menu.pop()

# {{{1 ASSEMBLY PARTS

menu.pop()
menu.append('&Parts')

toolset.registerKernelMenuButton(
        buttonText='|'.join(menu) + '|&Delete unused Parts',
        moduleName='assemblyMod',
        functionName='part_deleteUnused()',
        author='Carl Osterwisch',
        version=__version__,
        helpUrl=helpUrl,
        applicableModules=['Assembly'],
        description='Remove parts that are not referenced by any instances.')

class InstanceEditPicked(InstanceSelectProcedure):
        prompt = 'instance to edit part'
        method ='part_edit'

toolset.registerGuiMenuButton(
        buttonText='|'.join(menu) + '|&Edit Picked...',
        object=InstanceEditPicked(toolset),
        kernelInitString='import assemblyMod',
        author='Carl Osterwisch',
        version=__version__,
        helpUrl=helpUrl,
        applicableModules=['Assembly', 'Interaction', 'Load'],
        description='Graphically select instance to edit its part.'
        )

toolset.registerKernelMenuButton(
        buttonText='|'.join(menu) + '|&Instance unused Parts',
        moduleName='assemblyMod',
        functionName='part_instanceUnused()',
        author='Carl Osterwisch',
        version=__version__,
        helpUrl=helpUrl,
        applicableModules=['Assembly'],
        description='Instance parts that are not referenced by any instances.')

menu.append('&Mesh')

class InstanceMeshRefinePicked(InstanceSelectProcedure):
        prompt = 'instances of parts to refine global seed size'
        method ='part_meshRefine'

toolset.registerGuiMenuButton(
        buttonText='|'.join(menu) + '|&Refine Picked...',
        object=InstanceMeshRefinePicked(toolset),
        kernelInitString='import assemblyMod',
        author='Carl Osterwisch',
        version=__version__,
        helpUrl=helpUrl,
        applicableModules=['Assembly', 'Mesh'],
        description='Reduce global mesh seed size by factor 0.7 on selected Parts.')

toolset.registerKernelMenuButton(
        buttonText='|'.join(menu) + '|&Unmeshed Parts',
        moduleName='assemblyMod',
        functionName='part_meshUnmeshed()',
        author='Carl Osterwisch',
        version=__version__,
        helpUrl=helpUrl,
        applicableModules=['Assembly', 'Mesh'],
        description='Generate mesh on unmeshed used Parts and Instances.')

menu.pop()
menu.append('&Curve refinement')

class PartCurveRefinement(InstanceSelectProcedure):
        prompt = 'instances of parts to be refined'
        method = 'part_improveRefinement'

toolset.registerGuiMenuButton(
        buttonText='|'.join(menu) + '|&Improve for picked...',
        object=PartCurveRefinement(toolset),
        kernelInitString='import assemblyMod',
        author='Carl Osterwisch',
        version=__version__,
        helpUrl=helpUrl,
        applicableModules=['Assembly', 'Interaction', 'Load', 'Mesh'],
        description='Graphically select parts that need better curve refinement. ' \
            'The display refinement will improve each time selected until the maximum "extra fine" is achieved.'
        )

toolset.registerKernelMenuButton(
        buttonText='|'.join(menu) + '|&Reset all to coarse',
        moduleName='assemblyMod',
        functionName='part_resetRefinement()',
        author='Carl Osterwisch',
        version=__version__,
        helpUrl=helpUrl,
        applicableModules=['Assembly', 'Interaction', 'Load', 'Mesh'],
        description='Reset active parts to their default "coarse" level of geometry refinement. ' \
                'This setting may be required for section cuts.')

menu.pop()

# {{{1 PART module only

toolset.registerKernelMenuButton(
        buttonText='Principal mass properties',
        moduleName='assemblyMod',
        functionName='part_principalProperties()',
        author='Carl Osterwisch',
        version=__version__,
        helpUrl=helpUrl,
        applicableModules=['Part'],
        description='Calculate and report principal mass properties.')

toolset.registerKernelMenuButton(
        buttonText='Report surface areas',
        moduleName='assemblyMod',
        functionName='part_surfaceAreas()',
        author='Carl Osterwisch',
        version=__version__,
        helpUrl=helpUrl,
        applicableModules=['Part'],
        description='Calculate and report area of all named surfaces.')
