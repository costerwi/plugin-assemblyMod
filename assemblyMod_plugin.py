"""Define the AFXProcedure class to handle assembly modifications.

Carl Osterwisch <costerwi@gmail.com> November 2013
"""
__version__ = 0.1

from abaqusGui import *

###########################################################################
# Procedure definition
###########################################################################
class flatnessProcedure(AFXProcedure):
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
# Register the plugin
###########################################################################
toolset = getAFXApp().getAFXMainWindow().getPluginToolset()

toolset.registerGuiMenuButton(
        buttonText='&Delete intances...', 
        object=flatnessProcedure(toolset),
        kernelInitString='import assemblyMod',
        author='Carl Osterwisch',
        version=str(__version__),
        applicableModules=['Assembly'],
        description='Graphically select instances to remove from the assembly.'
        )

toolset.registerKernelMenuButton(
        buttonText='Delete &unused parts', 
        moduleName='assemblyMod',
        functionName='part_deleteUnused()',
        author='Carl Osterwisch',
        version=str(__version__),
        applicableModules=['Assembly'],
        description='Remove parts that are not referenced by any instances.')

