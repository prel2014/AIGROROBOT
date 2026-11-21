"""Render one recorded PhysX replica through its native eyes and exterior cameras."""
import argparse
import json
from pathlib import Path

import numpy as np

from film_field_cycle import Film, configure, encoder_path, eye_pixels, visibility


def render(env,source,out,index,ffmpeg):
    import torch
    from dexman.experiments import sha256
    from dexman.tasks.agriculture.harvest_route import SITES, binocular_visibility

    report = json.loads((source/"resultado.json").read_text(encoding="utf-8"))
    recorded = np.load(source/"poses_fisicas.npz",allow_pickle=False)
    rows = {step:values[index] for step,phase,values in json.loads((source/"mediciones.json").read_text())
            if phase not in ("recorrido","detenido")}
    trial = report["robots"][index]
    group, = env.batch.groups
    rover = group.rover.robot
    fruit = group.scene["fruit_0"]
    if list(recorded["joints"])!=list(rover.joint_names):
        raise ValueError("El orden articular grabado difiere del rover cargado")
    film = Film(out,[(trial["version"],int(trial["hand"]=="derecha"),None,None)],record_obstructions=True,
                robot_indices=[index])
    pixels,depth_rows,front_rows = [],[],[]
    sim = env.batch.sim
    overview,detail = group.scene["witness"],group.scene["detail"]
    labels = recorded["executed_phases"][:,index] if "executed_phases" in recorded else recorded["phases"]
    neighbor_poses = recorded.get("neighbor")
    for frame,(step,phase,state) in enumerate(zip(recorded["step"],recorded["phases"],recorded["poses"][:,index],strict=True)):
        q = torch.as_tensor(state.copy(),device=env.device)[None]
        root,angles,object_pose = q[:,:7].clone(),q[:,7:-7].contiguous(),q[:,-7:].clone()
        root[:,:3] += group.origins
        object_pose[:,:3] += group.origins
        rover.write_root_pose_to_sim_index(root_pose=root)
        rover.write_joint_position_to_sim_index(position=angles)
        fruit.write_root_pose_to_sim_index(root_pose=object_pose)
        if neighbor_poses is not None:
            pose = torch.as_tensor(neighbor_poses[frame,index].copy(),device=env.device)[None]
            pose[:,:3] += group.origins
            group.scene["neighbor"].write_root_pose_to_sim_index(root_pose=pose)
        sim.forward()
        group.scene.update(1./report["fps"])
        pos = root[:,:3]
        center = fruit.data.root_com_pos_w.torch
        def offset(x,y,z):
            return pos.new_tensor([x,y,z])
        focus = pos+offset(.26,0.,.46)
        overview.set_world_poses_from_view(pos+offset(-1.72,-1.45,2.95),focus)
        if phase in ("recorrido","detenido"):
            detail.set_world_poses_from_view(pos+offset(1.92,.50,1.82),focus)
        else:
            detail.set_world_poses_from_view(center+offset(.38,-.70,.43),center+offset(-.06,0.,-.04))
        sim.render()
        if frame==0:
            sim.render()
        lift,held = rows.get(int(step),(0.,0.))[:2]
        depth = film.frame(env.batch,overview,detail,str(labels[frame]),[lift],[held])
        count,front = eye_pixels(group,center)
        pixels.append([int(step),str(phase),count[0]])
        front_rows.append([int(step),str(phase),front[0]])
        depth_rows.append([int(step),str(phase),visibility(group,depth,center,SITES[report["crop"]].radius)[0]])
    film.encode(ffmpeg)
    (out/"pixeles.json").write_text(json.dumps(pixels))
    (out/"visibilidad.json").write_text(json.dumps(depth_rows))
    (out/"superficie_frente_al_fruto.json").write_text(json.dumps(front_rows))
    (out/"procedencia.json").write_text(json.dumps(dict(
        source=str(source),robot=index,frames=film.index,fps=report["fps"],trial=trial,
        binocular_visibility=binocular_visibility(pixels,depth_rows),
        states_sha256=sha256(source/"poses_fisicas.npz"),report_sha256=sha256(source/"resultado.json"),
        mode="Render RTX de poses físicas registradas; una imagen por muestra original."),ensure_ascii=False,indent=2),encoding="utf-8")


if __name__=="__main__":
    from isaaclab.app import add_launcher_args,launch_simulation
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--robot",required=True,type=int)
    parser.add_argument("--ffmpeg",default="ffmpeg")
    parser.add_argument("--eye-size",type=int,choices=(320,512,640),default=320,
                        help="Resolución cuadrada por ojo, conservando las ópticas y el campo de visión del rover")
    add_launcher_args(parser)
    args = parser.parse_args()
    source = args.source.resolve()
    report = json.loads((source/"resultado.json").read_text(encoding="utf-8"))
    if not 0<=args.robot<len(report["robots"]):
        parser.error("Índice de robot fuera del registro")
    cfg,_,_ = configure(source,report["crop"],[report["robots"][args.robot]["version"]],verify_eyes=True,
                        orange_task=report.get("orange_task"),order=report.get("instruction"),
                        rock_offset=report.get("rock_offset_m",.70))
    scene = getattr(cfg.scenes,report["crop"])
    for camera in (scene.left_eye,scene.right_eye):
        camera.height = camera.width = args.eye_size
    ffmpeg = encoder_path(args.ffmpeg)
    out = args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    with launch_simulation(cfg,args):
        import torch
        from isaaclab.sim import SimulationContext
        from dexman.tasks.agriculture.environment import AgriculturalPPOEnv
        with torch.inference_mode():
            env = AgriculturalPPOEnv(SimulationContext(cfg.sim),cfg,seed=report["seed"])
            render(env,source,out,args.robot,ffmpeg)
