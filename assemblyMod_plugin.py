"""Define the AFXProcedure class to handle assembly modifications.

Carl Osterwisch <costerwi@gmail.com> November 2013
"""
__version__ = 0.4

from abaqusGui import *

###########################################################################
# Procedure definition
###########################################################################
class instanceDeleteProcedure(AFXProcedure):
    "Class to start the planar query procedure"

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

    def getNextStep(self, previousStep):
        return None

###########################################################################
# Register the plugins
###########################################################################
toolset = getAFXApp().getAFXMainWindow().getPluginToolset()

toolset.registerGuiMenuButton(
        buttonText='&Delete intances...', 
        object=instanceDeleteProcedure(toolset),
        kernelInitString='import assemblyMod',
        author='Carl Osterwisch',
        version=str(__version__),
        applicableModules=['Assembly'],
        description='Graphically select instances to remove from the assembly.'
        )

toolset.registerKernelMenuButton(
        buttonText='Rename instances using part name', 
        moduleName='assemblyMod',
        functionName='instance_matchname()',
        author='Carl Osterwisch',
        version=str(__version__),
        applicableModules=['Assembly'],
        description='Update instance names using part name as a base.')

toolset.registerKernelMenuButton(
        buttonText='Delete &unused parts', 
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
        buttonText='Find &duplicate parts', 
        moduleName='assemblyMod',
        functionName='part_derefDuplicate()',
        author='Carl Osterwisch',
        version=str(__version__),
        applicableModules=['Assembly'],
        description='Unreference duplicate parts.')

toolset.registerKernelMenuButton(
        buttonText='Delete non-solid instances', 
        moduleName='assemblyMod',
        functionName='instance_delete_shell()',
        author='Carl Osterwisch',
        version=str(__version__),
        applicableModules=['Assembly'],
        description='Delete instances that have zero volume.')
