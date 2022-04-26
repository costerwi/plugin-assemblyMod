# plugin-assemblyMod
Abaqus/CAE plugins to work with Parts and their Instances

## Assembly module plugins

- Instances
  - **Edit Part...** - Graphically select instance to edit its part.
  - **Find duplicate parts** - Use mass properties to identify and replace instances of similar parts with multiple instances of one part.
  - **Reposition using 2 csys...** - Reposition instances based on selected source and destination csys.
  - **Rename using part name** - Update instance names using part name as a base.
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
  - **Instance unused Parts** - Instance parts that are not referenced by any instances.

## Part module plugins

- **Principal mass properties** - Calculate and report principal mass properties.
- **Report surface areas** - Calculate and report area of all named surfaces.
