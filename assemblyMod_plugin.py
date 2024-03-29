"""Define the AFXProcedure class to handle assembly modifications.

Carl Osterwisch <costerwi@gmail.com> November 2013
"""
__version__ = "0.9.1"

from abaqusGui import *

###########################################################################
# {{{1 Procedure definition
###########################################################################
class InstanceSelectProcedure(AFXProcedure):
    """Class to allow user to select Instances and run an assemblyMod command"""

    def __init__(self, owner, prompt, method, number=MANY):
        AFXProcedure.__init__(self, owner) # Construct the base class.
        self._prompt = prompt
        self._number = number

        # Command
        command = AFXGuiCommand(mode=self,
                method=method,
                objectName='assemblyMod',
                registerQuery=FALSE)

        # Keywords
        name = 'instance'
        if MANY == number:
            name += 's'
        self.instancesKw = AFXObjectKeyword(
                command=command,
                name=name,
                isRequired=TRUE)

    def getFirstStep(self):
        return AFXPickStep(
                owner=self,
                keyword=self.instancesKw,
                prompt='Select ' + self._prompt,
                entitiesToPick=INSTANCES,
                numberToPick=self._number,
                sequenceStyle=TUPLE)    # TUPLE or ARRAY


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
        version=str(__version__),
        helpUrl='https://github.com/costerwi/plugin-assemblyMod',
        applicableModules=['Assembly'],
        description='Use mass properties to automatically identify and replace instances of similar parts with multiple instances of one part. '
            'Two parts must have the same mass, area, and primary moments of inertia within 0.01% to be recognized as equal.'
        )

class InstanceDuplicatePicked(InstanceSelectProcedure):
    """CAE seems to register this class with the GuiMenuButton, not the instance of the class"""
    pass

toolset.registerGuiMenuButton(
        buttonText='|'.join(menu) + '|&Pick Duplicate Parts...',
        object=InstanceDuplicatePicked(toolset, 'instances to search for common parts', 'instance_derefDup'),
        kernelInitString='import assemblyMod',
        author='Carl Osterwisch',
        version=str(__version__),
        helpUrl='https://github.com/costerwi/plugin-assemblyMod',
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
        version=str(__version__),
        helpUrl='https://github.com/costerwi/plugin-assemblyMod',
        applicableModules=['Assembly'],
        description='Reposition instances based on position and orientation of selected source and destination csys.'
        )

toolset.registerKernelMenuButton(
        buttonText='|'.join(menu) + '|Rename using Part &name',
        moduleName='assemblyMod',
        functionName='instance_matchname()',
        author='Carl Osterwisch',
        version=str(__version__),
        helpUrl='https://github.com/costerwi/plugin-assemblyMod',
        applicableModules=['Assembly'],
        description='Update instance names using part name as a base.')

toolset.registerGuiMenuButton(
        buttonText='|'.join(menu) + '|&Rename using search/replace...',
        object=InstanceRenameProcedure(toolset),
        kernelInitString='import assemblyMod',
        author='Carl Osterwisch',
        version=str(__version__),
        helpUrl='https://github.com/costerwi/plugin-assemblyMod',
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
        version=str(__version__),
        helpUrl='https://github.com/costerwi/plugin-assemblyMod',
        applicableModules=['Assembly'],
        description='Delete instances that are currently hidden.')


toolset.registerKernelMenuButton(
        buttonText='|'.join(menu) + '|&Suppressed',
        moduleName='assemblyMod',
        functionName='instance_delete_suppressed()',
        author='Carl Osterwisch',
        version=str(__version__),
        helpUrl='https://github.com/costerwi/plugin-assemblyMod',
        applicableModules=['Assembly'],
        description='Delete instances that are currently suppressed.')

menu.pop()

# {{{1 ASSEMBLY INSTANCES SUPPRESS

menu.append('&Suppress')

class InstanceSuppressPicked(InstanceSelectProcedure):
    """CAE seems to register this class with the GuiMenuButton, not the instance of the class"""
    pass

toolset.registerGuiMenuButton(
        buttonText='|'.join(menu) + '|&Picked Instances...',
        object=InstanceSuppressPicked(toolset, 'instances to suppress', 'instance_suppress'),
        kernelInitString='import assemblyMod',
        author='Carl Osterwisch',
        version=str(__version__),
        helpUrl='https://github.com/costerwi/plugin-assemblyMod',
        applicableModules=['Assembly', 'Interaction', 'Load'],
        description='Graphically select instances to suppress in the assembly.'
        )

class InstanceSuppressPart(InstanceSelectProcedure):
    """CAE seems to register this class with the GuiMenuButton, not the instance of the class"""
    pass

toolset.registerGuiMenuButton(
        buttonText='|'.join(menu) + '|All Instances of &Part...',
        object=InstanceSuppressPart(toolset, 'parts to suppress', 'instance_suppress_part'),
        kernelInitString='import assemblyMod',
        author='Carl Osterwisch',
        version=str(__version__),
        helpUrl='https://github.com/costerwi/plugin-assemblyMod',
        applicableModules=['Assembly', 'Interaction', 'Load'],
        description='Select parts to suppress their instancse.'
        )

toolset.registerKernelMenuButton(
        buttonText='|'.join(menu) + '|Zero &Area Instances',
        moduleName='assemblyMod',
        functionName='instance_suppress_noArea()',
        author='Carl Osterwisch',
        version=str(__version__),
        helpUrl='https://github.com/costerwi/plugin-assemblyMod',
        applicableModules=['Assembly', 'Interaction', 'Load'],
        description='Suppress instances that have no surface area.')

toolset.registerKernelMenuButton(
        buttonText='|'.join(menu) + '|Zero &Volume Instances',
        moduleName='assemblyMod',
        functionName='instance_suppress_noVolume()',
        author='Carl Osterwisch',
        version=str(__version__),
        helpUrl='https://github.com/costerwi/plugin-assemblyMod',
        applicableModules=['Assembly', 'Interaction', 'Load'],
        description='Suppress instances that have no volume.')

toolset.registerKernelMenuButton(
        buttonText='|'.join(menu) + '|&Invert',
        moduleName='assemblyMod',
        functionName='instance_suppress_invert()',
        author='Carl Osterwisch',
        version=str(__version__),
        helpUrl='https://github.com/costerwi/plugin-assemblyMod',
        applicableModules=['Assembly', 'Interaction', 'Load'],
        description='Resume suppressed instances and suppress active instances.')

toolset.registerKernelMenuButton(
        buttonText='|'.join(menu) + '|&Resume all',
        moduleName='assemblyMod',
        functionName='instance_suppress_resumeAll()',
        author='Carl Osterwisch',
        version=str(__version__),
        helpUrl='https://github.com/costerwi/plugin-assemblyMod',
        applicableModules=['Assembly', 'Interaction', 'Load'],
        description='Resume suppressed instances.')

menu.pop()

# {{{1 ASSEMBLY INSTANCES HIDE

menu.append('&Hide')

class InstanceHidePicked(InstanceSelectProcedure):
    """CAE seems to register this class with the GuiMenuButton, not the instance of the class"""
    pass

toolset.registerGuiMenuButton(
        buttonText='|'.join(menu) + '|&Picked Instances...',
        object=InstanceHidePicked(toolset, 'instances to hide', 'instance_hide'),
        kernelInitString='import assemblyMod',
        author='Carl Osterwisch',
        version=str(__version__),
        helpUrl='https://github.com/costerwi/plugin-assemblyMod',
        applicableModules=['Assembly', 'Interaction', 'Load'],
        description='Graphically select instances to hide from the assembly.' \
                ' See also: right-click menu, Hide Instance.'
        )

class InstanceHidePart(InstanceSelectProcedure):
    """CAE seems to register this class with the GuiMenuButton, not the instance of the class"""
    pass

toolset.registerGuiMenuButton(
        buttonText='|'.join(menu) + '|All Instances of &Part...',
        object=InstanceHidePart(toolset, 'parts to hide', 'instance_hide_part'),
        kernelInitString='import assemblyMod',
        author='Carl Osterwisch',
        version=str(__version__),
        helpUrl='https://github.com/costerwi/plugin-assemblyMod',
        applicableModules=['Assembly', 'Interaction', 'Load'],
        description='Hide visible instances and show hidden instances.'
        )

toolset.registerKernelMenuButton(
        buttonText='|'.join(menu) + '|&Invert',
        moduleName='assemblyMod',
        functionName='instance_hide_invert()',
        author='Carl Osterwisch',
        version=str(__version__),
        helpUrl='https://github.com/costerwi/plugin-assemblyMod',
        applicableModules=['Assembly', 'Interaction', 'Load'],
        description='Show hidden instances and hide shown instances.')

toolset.registerKernelMenuButton(
        buttonText='|'.join(menu) + '|&Show all',
        moduleName='assemblyMod',
        functionName='instance_hide_showAll()',
        author='Carl Osterwisch',
        version=str(__version__),
        helpUrl='https://github.com/costerwi/plugin-assemblyMod',
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
        version=str(__version__),
        helpUrl='https://github.com/costerwi/plugin-assemblyMod',
        applicableModules=['Assembly'],
        description='Remove parts that are not referenced by any instances.')

class InstanceEditPicked(InstanceSelectProcedure):
    """CAE seems to register this class with the GuiMenuButton, not the instance of the class"""
    pass

toolset.registerGuiMenuButton(
        buttonText='|'.join(menu) + '|&Edit Picked...',
        object=InstanceEditPicked(toolset, 'instance to edit part', 'part_edit', ONE),
        kernelInitString='import assemblyMod',
        author='Carl Osterwisch',
        version=str(__version__),
        helpUrl='https://github.com/costerwi/plugin-assemblyMod',
        applicableModules=['Assembly', 'Interaction', 'Load'],
        description='Graphically select instance to edit its part.'
        )

toolset.registerKernelMenuButton(
        buttonText='|'.join(menu) + '|&Instance unused Parts',
        moduleName='assemblyMod',
        functionName='part_instanceUnused()',
        author='Carl Osterwisch',
        version=str(__version__),
        helpUrl='https://github.com/costerwi/plugin-assemblyMod',
        applicableModules=['Assembly'],
        description='Instance parts that are not referenced by any instances.')

toolset.registerKernelMenuButton(
        buttonText='|'.join(menu) + '|&Mesh unmeshed Parts',
        moduleName='assemblyMod',
        functionName='part_meshUsed()',
        author='Carl Osterwisch',
        version=str(__version__),
        helpUrl='https://github.com/costerwi/plugin-assemblyMod',
        applicableModules=['Assembly'],
        description='Generate mesh on unmeshed used Parts and Instances.')

# {{{1 PART

toolset.registerKernelMenuButton(
        buttonText='Principal mass properties',
        moduleName='assemblyMod',
        functionName='part_principalProperties()',
        author='Carl Osterwisch',
        version=str(__version__),
        helpUrl='https://github.com/costerwi/plugin-assemblyMod',
        applicableModules=['Part'],
        description='Calculate and report principal mass properties.')

toolset.registerKernelMenuButton(
        buttonText='Report surface areas',
        moduleName='assemblyMod',
        functionName='part_surfaceAreas()',
        author='Carl Osterwisch',
        version=str(__version__),
        helpUrl='https://github.com/costerwi/plugin-assemblyMod',
        applicableModules=['Part'],
        description='Calculate and report area of all named surfaces.')
