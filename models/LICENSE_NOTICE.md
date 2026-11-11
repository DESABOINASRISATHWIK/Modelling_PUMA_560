# Model provenance and licences

## What `puma560.xml` is
* **Kinematic tree** - written for this project by `tools/build_mjcf.py` from the PUMA-560 link dimensions in
  `src/robot_parameters.py`, using absolute zero-pose joint points/axes (no DH matrices), so that MuJoCo's forward
  kinematics is an independent check of the DH model.
* **Link meshes** (`meshes/link0..6.stl`) - the `UNIMATE/puma560` meshes distributed in the `rtb-data` PyPI package
  (v2.0.0, declared licence **MIT**, Peter Corke).  The same link meshes/parameters appear in **ARTE**
  (A Robotics Toolbox for Education, Arturo Gil, Univ. Miguel Hernandez, **LGPL-3.0**), which the Corke toolbox cites for
  this model.  I could not confirm the original authorship of the meshes from the package alone: if you publish or
  redistribute this repository, check upstream (github.com/4rtur1t0/ARTE) and keep the applicable notice.
  Link 1-6 files were converted from ASCII to binary STL (MuJoCo reads only binary STL); vertex coordinates changed by
  < 1e-8 m (float32 rounding only).
* **Masses / COM / inertia** - Armstrong, Khatib & Burdick (1986) as tabulated by Corke & Armstrong-Helouvry (1994).
  Two documented adjustments (needed for MuJoCo to accept the model): link 1 has no published mass (placeholder
  1 g, only its 0.35 kg m^2 inertia about the waist axis is meaningful); link 3's published inertia violates the
  triangle inequality (Iyy > Ixx + Izz) so its smallest moment is raised from 0.0125 to 0.0201 kg m^2.

## Why not MuJoCo Menagerie?
I did not find a PUMA-560 among the Menagerie arms.  The ROS `puma560_description` URDF (also bundled with rtb-data) was
examined but not used: it places the elbow offset a3 on the shoulder side (joint-3 and joint-4 axes intersect), which is
not the standard PUMA-560 geometry that the DH tables of the literature describe.
