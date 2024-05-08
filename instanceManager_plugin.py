from abaqusGui import *
import abaqusConstants

class myQuery:
    "Object used to register/unregister Queries"
    def __init__(self, object, subroutine):
        "register the query when this object is created"
        self.object = object
        self.subroutine = subroutine
        object.registerQuery(subroutine)
    def __del__(self):
        "unregister the query when this object is deleted"
        self.object.unregisterQuery(self.subroutine)


###########################################################################
# Dialog box
###########################################################################
class instanceManagerDB(AFXDataDialog):
    """The instance edit dialog box class

    editForm will create an instance of this class when the user requests it.
    """

    (
        ID_TABLE,
        ID_FILTER,
        ID_LAST
    ) = range(AFXDataDialog.ID_LAST, AFXDataDialog.ID_LAST + 3)


    def __init__(self, form):
        # Construct the base class.
        AFXDataDialog.__init__(self,
                mode=form,
                title="Instance Manager",
                opts=DIALOG_NORMAL|DECOR_RESIZE)

        mainframe = FXVerticalFrame(self, FRAME_SUNKEN | LAYOUT_FILL_X | LAYOUT_FILL_Y)

        self.table = AFXTable(
                p=mainframe,
                numVisRows=4,
                numVisColumns=7,
                numRows=1,
                numColumns=7,
                tgt=self,
                sel=self.ID_TABLE,
                opts=AFXTABLE_NORMAL|AFXTABLE_ROW_MODE)
        FXMAPFUNC(self, SEL_CLICKED, self.ID_TABLE, instanceManagerDB.onTable)
        FXMAPFUNC(self, SEL_COMMAND, self.ID_TABLE, instanceManagerDB.onCommand)
        self.table.setLeadingRows(numRows=1)
        self.table.setLeadingRowLabels('Name\tTx\tTy\tTz\tRx\tRy\tRz')
        self.table.setStretchableColumn(0) # Expand Name as necessary

        for col in range(self.table.getNumColumns()):
            self.table.setColumnSortable(col, TRUE) # All are sortable
            self.table.setColumnEditable(col, TRUE) # Allow edit
        self.table.setCurrentSortColumn(0) # Default sort by name

        self.table.setPopupOptions(AFXTable.POPUP_FILE)

        self.filter = ''  # Don't filter anything
        self.instancesQuery = None
        self.model = None
        AFXTextField(p=mainframe,
                ncols=15,
                labelText='Regular expression filter:',
                tgt=self,
                sel=self.ID_FILTER,
                opts=LAYOUT_FILL_X)
        FXMAPFUNC(self, SEL_COMMAND, self.ID_FILTER, instanceManagerDB.onFilter)


    def updateData(self):
        "Query the latest instance info"
        sendCommand("instanceManager.updateData()")


    def updateTable(self):
        "Read instance settings from customData.userInstances registered list"

        # Filter the data
        import re
        filterre = re.compile(self.filter, re.IGNORECASE)
        filtered = []
        for row in session.customData.instanceData:
            if filterre.search(row[0]):
                filtered.append( row )

        # Sort table data
        sortColumn = self.table.getCurrentSortColumn()
        filtered.sort(key=lambda a: a[sortColumn])
        if self.table.getColumnSortOrder(sortColumn) == AFXTable.SORT_DESCENDING:
            filtered.reverse()

        # Adjust table widget size
        diff = len(filtered) + 1 - self.table.getNumRows()
        if diff > 0:
            self.table.insertRows(
                    startRow=1,
                    numRows=diff,
                    notify=FALSE)
        elif diff < 0:
            self.table.deleteRows(
                    startRow=1,
                    numRows=-diff,
                    notify=FALSE)

        # Update table widget
        selected = self.getMode().selected
        for row, data in enumerate(filtered):
            tableRow = row + 1
            if data[0] == selected:
                selected = tableRow
            self.table.deselectRow(tableRow)
            for col, item in enumerate(data):
                self.table.setItemValue(
                        row=tableRow,
                        column=col,
                        valueText=str(item))

        if isinstance(selected, int):
            self.table.selectRow(selected)
            self.table.makeRowVisible(selected)


    def onCommand(self, sender, sel, ptr):
        " Called after edited table cell "
        row = sender.getCurrentRow()
        if row > 0:
            instance = self.getMode().selected
            col = sender.getCurrentColumn()
            value = sender.getItemValue(row, col)
            if not value:
                return
            try:
                if 0 == col:
                    sendCommand("assemblyMod.instance_rename(newNames={%r: %r})"%(
                        instance, value))
                    self.getMode().selected = newName
                elif col < 4:
                    position = [eval(sender.getItemValue(row, col)) for col in (1, 2, 3)]
                    sendCommand("assemblyMod.instance_moveTo(instance=%r, position=%r)"%(
                        instance, position) )
                else:
                    rotation = [eval(sender.getItemValue(row, col)) for col in (4, 5, 6)]
                    sendCommand("assemblyMod.instance_rotateTo(instance=%r, rotation=%r)"%(
                        instance, rotation) )
            except (SyntaxError, ValueError, NameError) as e:
                print(e)
                pass

        return 0


    def onTable(self, sender, sel, ptr):
        "Table was clicked - update the keyword or sorting"
        row = sender.getCurrentRow()
        if row > 0:
            id = sender.getItemValue(row, 0)
            self.getMode().selected = id
            sendCommand("instanceManager.outline(%r)"%id)
        if row == 0:
            self.updateTable()  # sorting has changed
        return 0


    def onFilter(self, sender, sel, ptr):
        "Search field was changed"
        self.filter = sender.getText()
        self.updateTable()
        return 0


    def onContextChange(self):
        "Called when viewport or displayed object are changed"
        context = getCurrentContext()
        mdbName, viewportName, objectPath, objectType, modelName, moduleName = context
        if self.model:
            self.model.rootAssembly.instances.unregisterQuery(self.updateData)
        self.model = mdb.models[context['modelName']]
        self.model.rootAssembly.instances.registerInclusive(self.updateData)


    def show(self):
        "Prepare to show the dialog box"
        # Register query and populate the table

        registerCurrentContext(self.onContextChange)
        session.customData.instanceData.registerQuery(self.updateTable)
        self.updateData()
        return AFXDataDialog.show(self)


    def hide(self):
        "Called to remove the dialog box"
        unregisterCurrentContext(self.onContextChange)
        if self.model:
            self.model.rootAssembly.instances.unregisterQuery(self.updateData)
        session.customData.instanceData.unregisterQuery(self.updateTable)
        return AFXDataDialog.hide(self)


###########################################################################
# Form definition
###########################################################################
class instanceManagerForm(AFXForm):
    "Class to launch the instances GUI"
    def __init__(self, owner):

        AFXForm.__init__(self, owner) # Construct the base class.
        self.selected = None

    def getFirstDialog(self):
        return instanceManagerDB(self)


###########################################################################
# Register abaqus plugin
###########################################################################
toolset = getAFXApp().getAFXMainWindow().getPluginToolset()

toolset.registerGuiMenuButton(
        buttonText='&Instances|&Manager...',
        object=instanceManagerForm(toolset),
        kernelInitString='import instanceManager; import assemblyMod',
        author='Carl Osterwisch',
        version="00",
        applicableModules=['Assembly'],
        description='Edit instance positions.')
