# plugin-assemblyMod
This is a collection of Abaqus/CAE plugins to help work with Parts and their Instances, especially in large imported assemblies.

### Assembly module plugins

- Instances
  - **Find duplicate parts** - Use mass properties to automatically identify and replace instances of similar parts with multiple instances of one part. Two parts must have the same mass, area, and primary moments of inertia within 0.01% to be recognized as equal.
  - **Pick Duplicate Parts...** - Graphically select instances to replace with a common part. Replacement will not happen if mass properties are significantly different from each other. Two parts must have the same mass, area, and primary moments of inertia within 1% to be recognized as equal.
  - **Reposition using 2 csys...** - Reposition instances based on selected source and destination csys. Similar to HM position command.
  - **Rename using part name** - Update instance names using their part name as the base. This also searches for an updates regions which refer to the renamed instances, such as Constraints.
  - **Rename using search/replace** - Update instance names using user specified search and replace strings or regular expressions.
  - Delete
    - **Hidden** - Delete instances that are currently hidden.
    - **Suppressed** - Delete instances that are currently suppressed.
  - Suppress
    - **All Instances of Part...** - Select parts to suppress their instances.
    - **Invert** - Resume suppressed instances and suppress active instances.
    - **Picked Instances...** - Graphically select instances to suppress.
    - **Resume all** - Resume suppressed instances.
    - **Zero Area Instances** - Suppress instances that have no surface area.
    - **Zero Volume Instances** - Suppress instances that have no volume.
  - Hide
    - **All Instances of Part...** - Select parts to hide their instances.
    - **Invert** - Show hidden instances and hide shown instances.
    - **Picked Instances...** - Graphically select instances to hide.
    - **Show all** - Show all hidden instances.
- Parts
  - **Delete unused Parts** - Remove parts that are not referenced by any instances.
  - **Edit Picked...** - Graphically select instance to edit its part.
  - **Instance unused Parts** - Instance parts that are not referenced by any instances.
  - **Mesh unmeshed Parts** - Generate mesh on unmeshed Parts and Instances used in the model.

### Part module plugins

- **Principal mass properties** - Calculate and report principal mass properties.
- **Report surface areas** - Calculate and report area of all named surfaces.

## Installation instructions

1. Download and unzip the [latest version](https://github.com/costerwi/plugin-assemblyMod/releases/latest)
2. Double-click the included `install.cmd` or manually copy files into your abaqus_plugins directory
3. Restart Abaqus CAE and you will find the above scripts in the Assembly module plug-ins menu
