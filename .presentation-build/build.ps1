$ErrorActionPreference = 'Stop'
$root = 'C:\Users\sahit\Downloads\PUMA560_Kinematics'
$pp = New-Object -ComObject PowerPoint.Application
$p = $pp.Presentations.Open('C:\Users\sahit\Downloads\Group_7_Robotics.pptx',0,0,0)
function RGB($r,$g,$b) { return [int]($r+256*$g+65536*$b) }
$navy = RGB 12 30 50
$white = RGB 240 244 249
$blue = RGB 83 163 228
$muted = RGB 175 194 213
function Text($s,$txt,$x,$y,$w,$h,$size=19,$color=$white,$bold=0) {
  $sh=$s.Shapes.AddTextbox(1,$x,$y,$w,$h)
  $sh.TextFrame.MarginLeft=0; $sh.TextFrame.MarginRight=0
  $sh.TextFrame.MarginTop=0; $sh.TextFrame.MarginBottom=0
  $sh.TextFrame.WordWrap=-1
  $sh.TextFrame.TextRange.Text=$txt
  $sh.TextFrame.TextRange.Font.Name='Questrial'
  $sh.TextFrame.TextRange.Font.Size=$size
  $sh.TextFrame.TextRange.Font.Color.RGB=$color
  $sh.TextFrame.TextRange.Font.Bold=[string]$bold
  $sh.TextFrame.TextRange.ParagraphFormat.SpaceAfter=10
  return $sh
}
function Slide($title,$source) {
  $s=$p.Slides.Add($p.Slides.Count+1,12)
  $s.FollowMasterBackground=0; $s.Background.Fill.Solid(); $s.Background.Fill.ForeColor.RGB=$navy
  $null=Text $s $title 32 24 655 50 29 $white -1
  $null=Text $s ('GROUP 7    PUMA 560                                       '+$s.SlideIndex) 32 380 655 14 9 $muted
  $s.NotesPage.Shapes.Placeholders.Item(2).TextFrame.TextRange.Text=$source
  return $s
}
function Picture($s,$file,$x,$y,$w,$h) {
  $path=Join-Path $root ('results\'+$file)
  $pic=$s.Shapes.AddPicture($path,0,-1,0,0,-1,-1)
  $ratio=[Math]::Min($w/$pic.Width,$h/$pic.Height)
  $pw=$pic.Width*$ratio; $ph=$pic.Height*$ratio; $pic.LockAspectRatio=0; $pic.Width=$pw; $pic.Height=$ph
  $pic.Left=$x+($w-$pic.Width)/2; $pic.Top=$y+($h-$pic.Height)/2
}
$s=Slide 'Project scope' 'Source: README.md, src/ and run_all.py. All analytical algorithms use project NumPy code. MuJoCo supplies visualization and an independent kinematic reference.'
$null=Text $s "A six revolute joint manipulator with a spherical wrist" 32 88 650 42 24 $blue
$null=Text $s "Robot specification and standard DH frames`r`nForward and analytical inverse kinematics`r`nVelocity kinematics and singularity analysis`r`nIndependent checks against MuJoCo" 32 150 650 190 22

$s=Slide 'Robot geometry and reference frames' 'Source: README.md, src/robot_parameters.py. Frame 0 equals the MuJoCo world. Joint i rotates about z_(i-1). The 0.04 m flange distance is a model choice cited to ARTE.'
$null=Text $s "O0: mounting plane`r`nO1: 0.67183 m above O0`r`n`r`nO4 = O5 = O6: wrist centre`r`nOE: 0.04 m along z6 from the wrist centre" 32 93 270 250 19
Picture $s 'fk_frames.png' 315 85 375 275

$s=Slide 'Standard Denavit-Hartenberg model' 'Source: README.md, src/robot_parameters.py and src/dh_model.py. Lengths in metres, angles in degrees in this table. All theta offsets are zero.'
$null=Text $s 'Ai = Rz(qi) Tz(di) Tx(ai) Rx(alpha_i)' 32 80 650 35 22 $blue
$table=$s.Shapes.AddTable(7,5,32,128,656,202).Table
$rows=@(@('Joint','theta','d (m)','a (m)','alpha (deg)'),@('1','q1','0.67183','0','90'),@('2','q2','0','0.4318','0'),@('3','q3','0.15005','0.0203','-90'),@('4','q4','0.4318','0','90'),@('5','q5','0','0','-90'),@('6','q6','0','0','0'))
for($i=1;$i -le 7;$i++){for($j=1;$j -le 5;$j++){$c=$table.Cell($i,$j).Shape; $c.TextFrame.TextRange.Text=$rows[$i-1][$j-1];$c.TextFrame.TextRange.Font.Name='Questrial';$c.TextFrame.TextRange.Font.Size=16;$c.TextFrame.TextRange.Font.Color.RGB=$white;$c.Fill.ForeColor.RGB=$navy;if($i -eq 1){$c.Fill.ForeColor.RGB=RGB 28 71 107}}}
$null=Text $s 'Flange transform: T6E = Trans(z, 0.04 m)' 32 342 650 26 18 $blue

$s=Slide 'Forward kinematics' 'Source: README.md and src/forward_kinematics.py. Figure: results/fk_mujoco.png. The pose contains both position and orientation.'
$null=Text $s "T0E = A1 A2 A3 A4 A5 A6 T6E`r`n`r`nInput: six joint angles`r`nOutput: end-effector pose, all DH frames and wrist centre`r`n`r`nThe MuJoCo pose provides an independent comparison" 32 90 290 266 19
Picture $s 'fk_mujoco.png' 338 85 350 278

$s=Slide 'Analytical inverse kinematics' 'Source: README.md and src/inverse_kinematics.py. Figure: results/ik_branches.png. Eight geometric branches apply to generic nonsingular reachable targets. Limits can exclude branches and singularities can merge them.'
$null=Text $s "1. Remove the flange offset`r`n2. Solve waist and elbow geometry`r`n3. Solve shoulder angle`r`n4. Extract wrist orientation`r`n5. Verify each branch with FK" 32 93 280 206 19
$null=Text $s 'Up to 8 branches: 2 waist x 2 elbow x 2 wrist' 32 312 280 50 20 $blue
Picture $s 'ik_branches.png' 322 86 370 276

$s=Slide 'Reachability and joint limits' 'Source: README.md, src/inverse_kinematics.py and results/verification_report.md. L = sqrt(a3^2 + d4^2), rho is the planar shoulder-to-wrist distance, and r is distance from the waist axis. The plotted workspace concerns the wrist centre.'
$null=Text $s "r >= |d3| = 0.15005 m`r`n|a2 - L| <= rho <= a2 + L`r`nL = sqrt(a3^2 + d4^2)`r`n`r`nOuter planar reach: 0.8641 m`r`n`r`nA reachable target can still violate every joint-limit branch" 32 88 278 275 18
Picture $s 'workspace.png' 324 84 369 278

$s=Slide 'Velocity kinematics' 'Source: README.md, src/jacobian.py and src/velocity_kinematics.py. Figure: results/velocity_ellipsoids.png. The geometric Jacobian is expressed in the base frame at the flange by default. Cross denotes the vector cross product.'
$null=Text $s "[v; omega] = J(q) qdot`r`n`r`nJv,i = z(i-1) cross (pE - p(i-1))`r`nJw,i = z(i-1)`r`n`r`nInverse velocity: pseudoinverse or damped least squares" 32 90 292 258 19
Picture $s 'velocity_ellipsoids.png' 337 88 350 270

$s=Slide 'Singularity conditions' 'Source: README.md and src/singularity_analysis.py. det(J) uses the wrist-centre reference point. The flange offset preserves the determinant. X = a2 cos(q2) + a3 cos(q2+q3) - d4 sin(q2+q3). phi = atan2(d4,a3) = approximately 87.3 degrees.'
$null=Text $s 'det(J) = a2 X (a3 sin(q3) + d4 cos(q3)) sin(q5)' 32 84 655 37 21 $blue
$null=Text $s "Shoulder: X = 0`r`nWrist centre reaches the cylinder about the waist axis`r`n`r`nElbow: a3 sin(q3) + d4 cos(q3) = 0`r`nStretched configuration within limits: q3 is about -87.3 deg`r`n`r`nWrist: sin(q5) = 0`r`nJoint 4 and joint 6 axes align. Within limits: q5 = 0 deg" 32 125 655 240 17

$s=Slide 'Singular configurations' 'Source: results/singularity_gallery.png, README.md and src/singularity_analysis.py. Each illustrated family has rank 5 in the reported verification. The full six-dimensional Jacobian defines loss of controllable twist.'
Picture $s 'singularity_gallery.png' 32 80 656 239
$null=Text $s 'The Jacobian loses rank and at least one twist direction becomes unavailable' 32 333 655 37 20 $blue

$s=Slide 'Near-singular joint speeds' 'Source: README.md, results/verification_report.md and results/joint_speed_vs_singularity.png. Values concern the weakest unit twist. Damping parameter lambda = 0.05 gives a 10 rad/s bound for that unit task. Singular values depend on units because J combines linear and angular rows.'
$null=Text $s "For the weakest twist:`r`n|qdot| = |twist| / sigma_min`r`n`r`nAt q5 = 0.0001 rad:`r`nPseudoinverse: about 37,500 rad/s`r`n`r`nDamping bounds joint speed at the cost of task error" 32 90 276 267 18
Picture $s 'joint_speed_vs_singularity.png' 326 88 363 273

$s=Slide 'Interactive MuJoCo demonstration' 'Source: README.md, run_all.py and src/simulation.py. Figure: results/velocity_mujoco.png. run_all.py uses smooth joint interpolation for visualization only, not physical control or a trajectory planner. Overlay calculations come from project code.'
$null=Text $s "Joint-angle or Cartesian input`r`n`r`nDH frames and IK branch overlays`r`n`r`nVelocity ellipsoid and arrows`r`n`r`nSingularity indicator" 32 94 260 250 20
Picture $s 'velocity_mujoco.png' 306 84 387 280

$s=Slide 'Numerical verification results' 'Source: results/verification_report.md and README.md. These are stored project report results, not measurements rerun during presentation creation. Fixed seeds and sample counts appear in that report.'
$table=$s.Shapes.AddTable(5,3,32,94,656,226).Table
$rows=@(@('Check','Sample size','Reported result'),@('FK vs MuJoCo','2,000 poses','Max position error 6.8e-16 m'),@('IK round trip','3,000 poses','3,000 / 3,000 recovered'),@('Jacobian vs differences','1,000 poses','Max absolute error 2.6e-10'),@('Determinant formula','Report check','Max difference 1.2e-16'))
$table.Columns.Item(1).Width=230;$table.Columns.Item(2).Width=130;$table.Columns.Item(3).Width=296
for($i=1;$i -le 5;$i++){for($j=1;$j -le 3;$j++){$c=$table.Cell($i,$j).Shape;$c.TextFrame.TextRange.Text=$rows[$i-1][$j-1];$c.TextFrame.TextRange.Font.Name='Questrial';$c.TextFrame.TextRange.Font.Size=15;$c.TextFrame.TextRange.Font.Color.RGB=$white;$c.Fill.ForeColor.RGB=$navy;if($i -eq 1){$c.Fill.ForeColor.RGB=RGB 28 71 107}}}
$null=Text $s 'Boundary, unreachable and singular targets also have dedicated checks' 32 338 650 30 18 $blue

$s=Slide 'Running the complete workflow' 'Source: run_all.py command documentation. --verify performs quick numerical checks. --no-sim prints the numerical workflow without opening the viewer. Joint inputs use degrees, positions use metres, Cartesian orientation uses degrees.'
$null=Text $s "Default pose and interactive demonstration`r`npython run_all.py`r`n`r`nChoose joint angles in degrees`r`npython run_all.py --q 30 50 -70 40 60 -20`r`n`r`nNumerical workflow with quick verification`r`npython run_all.py --no-sim --verify" 32 90 656 260 21

$s=Slide 'Conclusions and limitations' 'Source: README.md, run_all.py and results/verification_report.md. Dynamics, control, obstacle avoidance and trajectory planning remain outside this project. Viewer motion is visual interpolation.'
$null=Text $s "DH forward kinematics agrees with the MuJoCo reference`r`n`r`nAnalytical IK finds branches and diagnoses infeasible targets`r`n`r`nThe Jacobian explains lost motion and large joint speeds`r`n`r`nFuture work: dynamics, control and obstacle avoidance" 32 88 652 272 21

$s=Slide 'References and project sources' 'Primary project evidence: README.md; src/robot_parameters.py; src/forward_kinematics.py; src/inverse_kinematics.py; src/jacobian.py; src/singularity_analysis.py; results/verification_report.md; run_all.py; models/LICENSE_NOTICE.md. External model provenance follows the repository notices. ARTE: https://github.com/4rtur1t0/ARTE'
$null=Text $s "Project implementation and verification`r`nREADME.md, src/, run_all.py`r`nresults/verification_report.md`r`n`r`nRobot parameters`r`nCorke and Armstrong-Helouvry (1994)`r`nArmstrong, Khatib and Burdick (1986)`r`n`r`nARTE and rtb-data: models/LICENSE_NOTICE.md" 32 87 650 278 18

$out=Join-Path $root 'output\Group_7_PUMA560_Kinematics.pptx'
$p.SaveAs($out,24)
$render=Join-Path $root '.presentation-build\renders'
New-Item -ItemType Directory -Force $render | Out-Null
$issues=@()
foreach($s in $p.Slides){
  $s.Export((Join-Path $render ('slide-{0:D2}.png' -f $s.SlideIndex)),'PNG',1280,720)
  if($s.SlideIndex -gt 1){foreach($sh in $s.Shapes){if($sh.HasTextFrame -and $sh.TextFrame.HasText){if($sh.TextFrame.TextRange.BoundHeight -gt $sh.Height+3){$issues+=('Text overflow slide '+$s.SlideIndex)}}}}
}
Write-Output ('Saved '+$out+' Slides: '+$p.Slides.Count)
Write-Output $issues
$p.Close()



