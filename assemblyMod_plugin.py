"""Define the AFXProcedure class to handle assembly modifications.

Carl Osterwisch <costerwi@gmail.com> November 2013
"""
__version__ = "0.6.0"

from abaqusGui import *

###########################################################################
# Procedure definition
###########################################################################
class instanceDeleteProcedure(AFXProcedure):
    "Class to start the instance selection procedure"

    def __init__(self, owner):
        AFXProcedure.__init__(self, owner) # Construct the base class.

        # Command
        instance_delete = AFXGuiCommand(mode=self, 
                method='instance_delete', 
                objectName='assemblyMod',
                registerQuery=FALSE)

        # Keywords
        self.instancesKw = AFXObjectKeyword(
                command=instance_delete,
                name='instances',
                isRequired=TRUE)

    def getFirstStep(self):
        return AFXPickStep(
                owner=self,
                keyword=self.instancesKw,
                prompt="Select instances to delete",
                entitiesToPick=INSTANCES,
                numberToPick=MANY,
                sequenceStyle=TUPLE)    # TUPLE or ARRAY


class instanceHideProcedure(AFXProcedure):
    "Class to start the instance selection procedure"

    def __init__(self, owner):
        AFXProcedure.__init__(self, owner) # Construct the base class.

        # Command
        instance_cmd = AFXGuiCommand(mode=self,
                method='instance_hideUnselected',
                objectName='assemblyMod',
                registerQuery=FALSE)

        # Keywords
        self.instancesKw = AFXObjectKeyword(
                command=instance_cmd,
                name='instances',
                isRequired=TRUE)

    def getFirstStep(self):
        return AFXPickStep(
                owner=self,
                keyword=self.instancesKw,
                prompt="Select instances to keep visible",
                entitiesToPick=INSTANCES,
                numberToPick=MANY,
                sequenceStyle=TUPLE)    # TUPLE or ARRAY


class instanceRepositionProcedure(AFXProcedure):
    "Class to start the instance selection procedure"

    def __init__(self, owner):
        AFXProcedure.__init__(self, owner) # Construct the base class.

        # Command
        instance_cmd = AFXGuiCommand(mode=self,
                method='instance_reposition',
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


###########################################################################
# Register the plugins
###########################################################################
toolset = getAFXApp().getAFXMainWindow().getPluginToolset()

toolset.registerGuiMenuButton(
        buttonText='&Instances|&Delete...',
        object=instanceDeleteProcedure(toolset),
        kernelInitString='import assemblyMod',
        author='Carl Osterwisch',
        version=str(__version__),
        applicableModules=['Assembly'],
        description='Graphically select instances to remove from the assembly.'
        )

toolset.registerGuiMenuButton(
        buttonText='&Instances|&Hide unselected...',
        object=instanceHideProcedure(toolset),
        kernelInitString='import assemblyMod',
        author='Carl Osterwisch',
        version=str(__version__),
        applicableModules=['Assembly'],
        description='Graphically select instances to keep visible.'
        )

toolset.registerGuiMenuButton(
        buttonText='&Instances|&Reposition using 2 csys...',
        object=instanceRepositionProcedure(toolset),
        kernelInitString='import assemblyMod',
        author='Carl Osterwisch',
        version=str(__version__),
        applicableModules=['Assembly'],
        description='Reposition instances based on selected source and destination csys.'
        )

toolset.registerKernelMenuButton(
        buttonText='&Instances|&Rename using part name',
        moduleName='assemblyMod',
        functionName='instance_matchname()',
        author='Carl Osterwisch',
        version=str(__version__),
        applicableModules=['Assembly'],
        description='Update instance names using part name as a base.')

toolset.registerKernelMenuButton(
        buttonText='&Parts|Delete &unused',
        moduleName='assemblyMod',
        functionName='part_deleteUnused()',
        author='Carl Osterwisch',
        version=str(__version__),
        applicableModules=['Assembly'],
        description='Remove parts that are not referenced by any instances.')

toolset.registerKernelMenuButton(
        buttonText='Principal mass properties', 
        moduleName='assemblyMod',
        functionName='part_principalProperties()',
        author='Carl Osterwisch',
        version=str(__version__),
        applicableModules=['Part'],
        description='Calculate and report principal mass properties.')

toolset.registerKernelMenuButton(
        buttonText='Report surface areas', 
        moduleName='assemblyMod',
        functionName='part_surfaceAreas()',
        author='Carl Osterwisch',
        version=str(__version__),
        applicableModules=['Part'],
        description='Calculate and report area of all named surfaces.')

toolset.registerKernelMenuButton(
        buttonText='&Instances|Find &duplicate parts',
        moduleName='assemblyMod',
        functionName='part_derefDuplicate()',
        author='Carl Osterwisch',
        version=str(__version__),
        applicableModules=['Assembly'],
        description='Unreference duplicate parts.')

toolset.registerKernelMenuButton(
        buttonText='&Instances|Delete &hollow',
        moduleName='assemblyMod',
        functionName='instance_delete_hollow()',
        author='Carl Osterwisch',
        version=str(__version__),
        applicableModules=['Assembly'],
        description='Delete instances that have zero volume.')
