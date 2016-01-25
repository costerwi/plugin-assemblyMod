"""Define the AFXProcedure class to handle assembly modifications.

Carl Osterwisch <costerwi@gmail.com> January 2016
"""
__version__ = 0.1

from abaqusGui import *

###########################################################################
# Procedure definition
###########################################################################
class instanceEditPartProcedure(AFXProcedure):
    "Class to start the instance selection procedure"

    def __init__(self, owner):
        AFXProcedure.__init__(self, owner) # Construct the base class.

        # Command
        cmd = AFXGuiCommand(mode=self,
                method='instance_editPart',
                objectName='assemblyMod',
                registerQuery=FALSE)

        # Keywords
        self.instancesKw = AFXObjectKeyword(
                command=cmd,
                name='instance',
                isRequired=TRUE)

    def getFirstStep(self):
        return AFXPickStep(
                owner=self,
                keyword=self.instancesKw,
                prompt="Select instance to edit part",
                entitiesToPick=INSTANCES,
                numberToPick=ONE,
                sequenceStyle=TUPLE)    # TUPLE or ARRAY

###########################################################################
# Register the plugins
###########################################################################
toolset = getAFXApp().getAFXMainWindow().getPluginToolset()

toolset.registerGuiMenuButton(
        buttonText='&Instances|Edit &part...',
        object=instanceEditPartProcedure(toolset),
        kernelInitString='import assemblyMod',
        author='Carl Osterwisch',
        version=str(__version__),
        applicableModules=['Assembly'],
        description='Graphically select instance to edit its part.'
        )

