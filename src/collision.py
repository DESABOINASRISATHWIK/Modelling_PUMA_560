import numpy as np
import mujoco
from . import robot_parameters as rp
from .scene import vector

class CollisionChecker:
    def __init__(self,model):
        self.model=model
        self.data=mujoco.MjData(model)
        self.joints=np.array([model.joint(n).id for n in rp.MJCF_JOINT_NAMES])
        self.qadr=model.jnt_qposadr[self.joints]
        self.geom_names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) or f"geom_{i}" for i in range(model.ngeom)]
        self.geom_ids={n:i for i,n in enumerate(self.geom_names)}
        self.robot_geoms={n for n in self.geom_names if n.startswith("collision_link")}
        self.obstacle_geoms={n for n in self.geom_names if n.startswith("obstacle_")}

    def _set_configuration(self,q):
        q=vector(q,6,"q")
        mujoco.mj_resetData(self.model,self.data)
        self.data.qpos[self.qadr]=q
        mujoco.mj_forward(self.model,self.data)

    def contacts(self,q):
        self._set_configuration(q)
        return [(self.geom_names[c.geom1],self.geom_names[c.geom2]) for c in self.data.contact if c.dist<=0]

    def classify_contacts(self,q):
        result={"obstacle":[],"self_collision":[],"floor":[],"other":[]}
        for a,b in self.contacts(q):
            if a.startswith("obstacle_") or b.startswith("obstacle_"): result["obstacle"].append((a,b))
            elif a=="floor" or b=="floor": result["floor"].append((a,b))
            elif a in self.robot_geoms and b in self.robot_geoms: result["self_collision"].append((a,b))
            else: result["other"].append((a,b))
        return result

    def is_obstacle_free(self,q):
        return not self.classify_contacts(q)["obstacle"]

    def is_configuration_valid(self,q):
        q=vector(q,6,"q")
        limits=self.model.jnt_range[self.joints]
        return bool(np.all(q>=limits[:,0]) and np.all(q<=limits[:,1]) and not self.contacts(q))

    def is_edge_valid(self,start,end,resolution=0.02):
        start=vector(start,6,"start")
        end=vector(end,6,"end")
        if not np.isfinite(resolution) or resolution<=0: raise ValueError("resolution must be positive")
        steps=max(1,int(np.ceil(np.max(np.abs(end-start))/resolution)))
        for t in np.linspace(0,1,steps+1):
            if not self.is_configuration_valid(start+t*(end-start)): return False
        return True

    def validate_path(self,path,resolution=0.02):
        path=np.asarray(path,dtype=float)
        if path.ndim!=2 or path.shape[1]!=6 or len(path)==0 or not np.all(np.isfinite(path)): raise ValueError("path must have shape (N,6) and finite values")
        if len(path)==1: return self.is_configuration_valid(path[0])
        return all(self.is_edge_valid(path[i],path[i+1],resolution) for i in range(len(path)-1))

    def obstacle_clearance(self,q,max_distance=1.0):
        if max_distance<=0: raise ValueError("max_distance must be positive")
        self._set_configuration(q)
        robot_ids=[self.geom_ids[n] for n in self.robot_geoms]
        obstacle_ids=[self.geom_ids[n] for n in self.obstacle_geoms]
        if not robot_ids or not obstacle_ids: return float(max_distance)
        minimum=float(max_distance)
        fromto=np.zeros(6)
        for r in robot_ids:
            for o in obstacle_ids:
                d=mujoco.mj_geomDistance(self.model,self.data,r,o,float(max_distance),fromto)
                minimum=min(minimum,float(d))
        return minimum