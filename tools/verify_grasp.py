"""Verify a physical scripted grasp from an IK reset pose, without attaching the object."""
import argparse
import json
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tilt", type=float, default=45.)
    parser.add_argument("--height", type=float, default=.11)
    parser.add_argument("--thumb-sweep", action=argparse.BooleanOptionalAction, default=True)
    add_launcher_args(parser)
    args = parser.parse_args()
    from dexman.tasks.lift.config import LiftEnvCfg
    cfg = LiftEnvCfg(); cfg.scene.num_envs = 16; cfg.episode_length_s = 30; cfg.arm_half_range = 3.2
    with launch_simulation(cfg, args):
        import numpy as np
        import torch
        import gymnasium as gym
        import omni.usd
        from pxr import Gf, Usd, UsdPhysics
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation
        import dexman.tasks  # noqa
        from dexman.robot.joints import ARM, JOINT_NAMES
        env = gym.make("Dexman-UR10e-Shadow-Lift-v0", cfg=cfg)
        raw = env.unwrapped
        try:
            with torch.inference_mode():
                env.reset()
                base=raw.home_action.repeat(16,1)
                for _ in range(10): env.step(base)
                stage=omni.usd.get_context().get_stage()
                root="/World/envs/env_0/Robot"
                parents={}
                for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
                    if prim.IsA(UsdPhysics.Joint):
                        joint=UsdPhysics.Joint(prim)
                        a,b=joint.GetBody0Rel().GetTargets(),joint.GetBody1Rel().GetTargets()
                        if a and b: parents[str(b[0])]=(str(a[0]),joint)
                chain=[];current=root+"/Hand/robot0_palm"
                base_path=root+"/Arm/"+raw.robot.body_names[0]
                def frame(pos,rot):
                    m=Gf.Matrix4d(1);m.SetRotate(Gf.Quatd(rot));m.SetTranslateOnly(Gf.Vec3d(pos))
                    return np.array(m).T
                while current!=base_path:
                    parent,j=parents[current]
                    axis=j.GetPrim().GetAttribute("physics:axis").Get() or "X"
                    chain.append((j.GetPrim().GetName(),frame(j.GetLocalPos0Attr().Get(),j.GetLocalRot0Attr().Get()),
                        np.linalg.inv(frame(j.GetLocalPos1Attr().Get(),j.GetLocalRot1Attr().Get())),axis))
                    current=parent
                chain.reverse()
                origin=raw.scene.env_origins[0].cpu().numpy()
                root_mat=np.eye(4)
                root_mat[:3,:3]=Rotation.from_quat(raw.robot.data.body_quat_w.torch[0,0].cpu()).as_matrix()
                root_mat[:3,3]=raw.robot.data.body_pos_w.torch[0,0].cpu().numpy()-origin
                fixed_q=dict(zip(raw.robot.joint_names,raw.robot.data.joint_pos.torch[0].cpu().tolist()))
                def fk(q):
                    m=root_mat.copy()
                    for name,a,b,axis in chain:
                        angle=q[ARM.index(name)] if name in ARM else fixed_q.get(name,0.)
                        vec=np.zeros(3);vec["XYZ".index(axis)]=angle
                        turn=np.eye(4);turn[:3,:3]=Rotation.from_rotvec(vec).as_matrix()
                        m=m@a@turn@b
                    return m
                initial=raw.robot.data.joint_pos.torch[0,raw.joint_ids[:6]].cpu().numpy()
                actual=raw.robot.data.body_pos_w.torch[0,raw.palm_id].cpu().numpy()-origin
                fk_error=float(np.linalg.norm(fk(initial)[:3,3]-actual))
                print("FK",fk(initial)[:3,3].tolist(),actual.tolist(),"error",fk_error,flush=True)
                assert fk_error<1e-4, "FK does not match the simulator"
                desired_rot=Rotation.from_euler('z',180,degrees=True)*Rotation.from_euler('y',args.tilt,degrees=True)*Rotation.from_matrix(fk(initial)[:3,:3])
                cubes=(raw.object.data.root_pos_w.torch-raw.scene.env_origins).cpu().numpy()
                offsets=np.array([[x,y,args.height] for x in (-.01,-.03,-.05,-.07) for y in (0.,-.02,-.04,-.06)])
                if args.thumb_sweep: offsets[:]=[-.07,-.04,args.height]
                solutions=[];lift_solutions=[];errors=[]
                rng=np.random.default_rng(7)
                def solve(target,seed):
                    def residual(q):
                        m=fk(q)
                        return np.r_[10*(m[:3,3]-target),(desired_rot*Rotation.from_matrix(m[:3,:3]).inv()).as_rotvec()]
                    best=None
                    for trial in range(5):
                        start=seed if trial==0 else initial+rng.normal(0,1,6)
                        result=least_squares(residual,np.clip(start,-6.2,6.2),bounds=(-6.28,6.28),max_nfev=250)
                        if best is None or np.linalg.norm(result.fun)<np.linalg.norm(best.fun):best=result
                        if np.linalg.norm(best.fun)<1e-5:break
                    home=np.array([0.,-1.5708,1.5708,-1.5708,-1.5708,0.])
                    wrapped=home+np.arctan2(np.sin(best.x-home),np.cos(best.x-home))
                    return wrapped,float(np.linalg.norm(best.fun))
                for i in range(16):
                    q,error=solve(cubes[i]+offsets[i],initial)
                    qlift,_=solve(cubes[i]+offsets[i]+[0,0,.16],q)
                    solutions.append(q);lift_solutions.append(qlift);errors.append(error)
                print("IK errors",errors,flush=True)
                qstart=raw.robot.data.default_joint_pos.torch.clone()
                for name in ("robot0_WRJ1","robot0_WRJ0"):
                    qstart[:,raw.robot.joint_names.index(name)]=fixed_q[name]
                qstart[:,raw.joint_ids[:6]]=torch.tensor(np.array(solutions),device=raw.device,dtype=torch.float32)
                raw.robot.write_joint_position_to_sim_index(position=qstart)
                raw.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(qstart))
                raw.targets.copy_(qstart[:,raw.joint_ids])
                desired=qstart[:,raw.joint_ids].clone()
                def encode(q):
                    f=2*(q-raw.limits[:,0])/(raw.limits[:,1]-raw.limits[:,0])-1
                    return torch.cat((f[:,:22],f[:,22::2]),-1)
                closed=desired.clone()
                for finger in ("FF","MF","RF","LF"):
                    for j,angle in [(2,.9),(1,.8),(0,.8)]: closed[:,JOINT_NAMES.index(f"robot0_{finger}J{j}")]=angle
                for name,angle in [("robot0_THJ4",.4),("robot0_THJ3",.7),("robot0_THJ0",-.8)]:closed[:,JOINT_NAMES.index(name)]=angle
                if args.thumb_sweep:
                    closed[:,JOINT_NAMES.index("robot0_THJ3")]=torch.tensor([a for a in (.4,.7,1.,1.2) for b in range(4)],device=raw.device)
                    closed[:,JOINT_NAMES.index("robot0_THJ0")]=torch.tensor([b for a in range(4) for b in (-.4,-.8,-1.2,-1.5)],device=raw.device)
                max_lift=torch.zeros(16,device=raw.device)
                for step in range(300):
                    command=desired.lerp(closed,min(max((step-30)/60,0),1))
                    if step>=120:
                        arm=torch.tensor(np.array(solutions),device=raw.device,dtype=torch.float32).lerp(
                            torch.tensor(np.array(lift_solutions),device=raw.device,dtype=torch.float32),min((step-120)/90,1))
                        command[:,:6]=arm
                    raw.robot.actuators.target_command.set_effort_index(value=raw.robot.data.gravity_compensation_forces.torch)
                    env.step(encode(command))
                    lift=raw.object.data.root_pos_w.torch[:,2]-raw.scene.env_origins[:,2]-.625
                    max_lift=torch.maximum(max_lift,lift)
                    if step in (29,119,209,299):
                        print(step,"lift",lift.cpu().tolist(),flush=True)
                        print("palm offsets",(raw.robot.data.body_pos_w.torch[:,raw.palm_id]-raw.object.data.root_pos_w.torch).cpu().tolist(),flush=True)
                        print("arm q",raw.robot.data.joint_pos.torch[0,raw.joint_ids[:6]].cpu().tolist(),"target",command[0,:6].cpu().tolist(),flush=True)
                        print("tips offsets",(raw.robot.data.body_pos_w.torch[0,raw.tip_ids]-raw.object.data.root_pos_w.torch[0]).cpu().tolist(),flush=True)
                report={"ik_errors":errors,"arm_q":np.array(solutions).tolist(),"lift_arm_q":np.array(lift_solutions).tolist(),
                    "cube_local":cubes.tolist(),"offsets":offsets.tolist(),"max_lift_m":max_lift.cpu().tolist(),
                    "final_lift_m":lift.cpu().tolist(),"closed_q":closed.cpu().tolist()}
                report["stable_lift_count"] = int((lift > .14).sum().item())
                out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+"\n")
                print("Saved",out,flush=True)
                if not report["stable_lift_count"]:
                    raise RuntimeError("No candidate maintained the physical lift; inspect the calibration report")
        finally:env.close()


if __name__=="__main__":main()
