"""Build an offline, synchronized viewer from completed field films and verify both streams."""
import argparse
import json
from pathlib import Path
import subprocess


LABELS = {"citrus":"Naranjal","banana":"Platanal","vineyard":"Viñedo","berries":"Arandanal"}


def build(folder,ffprobe):
    clips = []
    for path in sorted(folder.glob("*/resultado.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("crop") not in LABELS:
            continue
        phases = {}
        rows = json.loads((path.parent/"visibilidad.json").read_text())
        for frame,(_,phase,_) in enumerate(rows):
            phases.setdefault(phase,frame/report["fps"])
        for robot in report["robots"]:
            stem = f"robot_{robot['robot']:03}"
            pair = [path.parent/(stem+k+".mp4") for k in ("","_ojos")]
            for file in pair:
                probe = subprocess.check_output([ffprobe,"-v","error","-select_streams","v:0","-count_frames",
                    "-show_entries","stream=codec_name,r_frame_rate,nb_read_frames","-of","json",str(file)])
                stream, = json.loads(probe)["streams"]
                if (stream["codec_name"],stream["r_frame_rate"],int(stream["nb_read_frames"]))!=(
                    "h264",f"{report['fps']}/1",report["frames"]):
                    raise ValueError(f"Incorrect frame timing or incomplete video: {file}")
            quality = [v[robot["robot"]] for _,phase,v in rows if phase in ("aproximacion","cierre")]
            visible = round(100*sum("linea_libre" in row for row in quality)/len(quality))
            profile = " · elevación ampliada" if report.get("profile")=="Elevación ampliada" else ""
            clips.append(dict(label=f"{LABELS[report['crop']]} · {robot['version']} · mano {robot['hand']}{profile}",
                videos=[file.relative_to(folder).as_posix() for file in pair],phases=phases,distance=robot["travel_m"],
                grip=robot["grasp"],hold=robot["hold_15cm"],deposit=robot["deposit"],peak=round(100*robot["peak_lift_m"],1),
                visibility=visible,
                report=path.relative_to(folder).as_posix()))
    if not clips:
        raise ValueError("No hay grabaciones completas")
    page = """<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cosecha en los surcos</title><style>
*{box-sizing:border-box}body{font:16px system-ui;margin:auto;max-width:1500px;padding:24px;background:#15211c;color:#edf4eb}
select,button,input{font:inherit}button,select{background:#e3edda;border:0;border-radius:5px;padding:7px 11px;margin:3px;color:#15211c;cursor:pointer}
main{display:grid;grid-template-columns:1.4fr 1fr;gap:16px}video{width:100%;max-height:67vh;background:#0a100c}figure{margin:0}figcaption{margin:10px 0}
a{color:#a7d9ae}p{line-height:1.5}#seek{width:min(65%,800px)}@media(max-width:730px){main{grid-template-columns:1fr}}
</style><h1>Cosecha en los surcos</h1><label>Ensayo <select id="trial"></select></label>
<main><figure><figcaption>Vista exterior y detalle</figcaption><video id="ext" muted playsinline preload="metadata"></video></figure>
<figure><figcaption>Ojos del rover · RGB arriba y profundidad abajo</figcaption><video id="eyes" muted playsinline preload="metadata"></video></figure></main>
<p><button id="play">Reproducir</button> <input type="range" id="seek" min="0" step=".01" value="0" aria-label="Tiempo"><span id="time"></span></p>
<nav aria-label="Ir a una fase" id="phases"></nav><p id="info"></p><p id="vision"></p><p id="files"></p>
<script>const clips=CLIPDATA; const el=id=>document.getElementById(id);
const ext=el('ext'), eyes=el('eyes'), seek=el('seek'), trial=el('trial');
clips.forEach((c,i)=>trial.add(new Option(c.label,i)));
function jump(t){for(const v of [ext,eyes])if(v.readyState)v.currentTime=t;seek.value=t}
function select(){ext.pause();eyes.pause();const c=clips[trial.selectedIndex];ext.src=c.videos[0];eyes.src=c.videos[1];seek.value=0;
el('phases').replaceChildren();Object.entries(c.phases).forEach(([name,t])=>{let b=document.createElement('button');b.textContent=name;b.onclick=()=>jump(t);el('phases').append(b)});
const yes=b=>b?'sí':'no';el('info').textContent='Recorrido '+c.distance+' m · Agarre confirmado '+yes(c.grip)+' · Sostén a 15 cm durante 0,5 s '+yes(c.hold)+' · Depósito '+yes(c.deposit)+' · Elevación máxima '+c.peak+' cm.';
el('vision').textContent='Aproximación y cierre · Al menos un ojo tuvo la línea al fruto despejada según profundidad en el '+c.visibility+' % de los fotogramas.';
el('files').replaceChildren();[...c.videos,c.report].forEach((f,i)=>{const a=document.createElement('a');a.href=f;a.textContent=['Abrir MP4 exterior','Abrir MP4 de los ojos','Mediciones del ensayo'][i];el('files').append(a,'　')});}
trial.onchange=select; select();seek.oninput=()=>jump(+seek.value);
ext.onloadedmetadata=()=>{seek.max=ext.duration}; ext.ontimeupdate=()=>{seek.value=ext.currentTime;el('time').textContent=ext.currentTime.toFixed(1)+' s';
if(!ext.paused&&eyes.readyState>=2&&Math.abs(eyes.currentTime-ext.currentTime)>.12)eyes.currentTime=ext.currentTime};
el('play').onclick=async()=>{if(!ext.paused){ext.pause();eyes.pause();return}if(ext.ended)jump(0);eyes.currentTime=ext.currentTime;await Promise.all([ext.play(),eyes.play()])};
ext.onpause=()=>{eyes.pause();el('play').textContent='Reproducir'};ext.onplay=()=>el('play').textContent='Pausar';
</script></html>"""
    (folder/"index.html").write_text(page.replace("CLIPDATA",json.dumps(clips,ensure_ascii=False)),encoding="utf-8")
    print(f"{len(clips)} pares verificados en {folder/'index.html'}")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder",type=Path)
    parser.add_argument("--ffprobe",default="ffprobe")
    args = parser.parse_args()
    build(args.folder,args.ffprobe)
