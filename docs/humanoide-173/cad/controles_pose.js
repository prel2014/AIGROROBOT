// Fuente del módulo de CAD Viewer. preparar_controles.py incorpora los datos de las manos y los cables.
const parameters = {};
const features = {};
const sides = [
  {key: "der", side: 1, arm: 10, leg: 9, kneeMotor: 12, elbowMotor: 14, branch: 3},
  {key: "izq", side: -1, arm: 8, leg: 7, kneeMotor: 11, elbowMotor: 13, branch: 2},
];
function parameter(id, label, min, max, unit = "°", step = 1) {
  parameters[id] = {type: "number", label, min, max, unit, step, default: 0};
}
function feature(id, paths) {
  features[id] = {ref: "#" + paths.map(p => "o1." + p).join(",")};
}
parameter("cuello", "Giro del cuello", -40, 40);
parameters.carcasa_piernas = {type:"boolean",label:"Carcasa exterior de las piernas",default:true};
parameters.carcasa_brazos = {type:"boolean",label:"Carcasa exterior de los brazos",default:true};
feature("carenados_brazos", sides.flatMap(({arm,elbowMotor})=>
  [arm+".3.5",arm+".3.6",arm+".1.4",elbowMotor+".1.3"]));
feature("carenados_piernas", sides.flatMap(({leg,kneeMotor})=>[leg+".3.35",leg+".5.35",leg+".5.37",kneeMotor+".1.3"]));
feature("cabeza", ["6", "1.15"]);
for (const {key, side, arm, leg, kneeMotor, elbowMotor, branch} of sides) {
  const label = side > 0 ? "D" : "I";
  for (const [name, title, low, high] of [
    ["hombro", "Hombro", -20, 65], ["codo", "Codo", 0, 90],
    ["muneca", "Muñeca · ajuste", -5, 25], ["cadera", "Cadera", -15, 35],
    ["rodilla", "Rodilla", 0, 80], ["tobillo", "Tobillo", -12, 12],
  ]) parameter(name + "_" + key, title + " " + label, low, high);
  parameter("cierre_" + key, "Cierre de mano " + label, 0, 100, "%", 5);
  const wire = "16." + branch;
  const traction = "19." + (side>0 ? 2 : 1);
  feature("can_motor_"+key,[traction+".2"]);
  feature("potencia_motor_"+key,[traction+".3"]);
  feature("brazo_" + key, [arm, elbowMotor, traction+".4", wire+".12", wire+".13", ...[4,8,9,10,11].map(i => wire+"."+i)]);
  feature("antebrazo_" + key, [
    ...[1,"2.3","2.4",6,7,8,9].map(i => arm+"."+i), elbowMotor+".3",
    ...[4,8,9,10,11].map(i => wire+"."+i),
  ]);
  feature("mano_" + key, [arm+".8"]);
  feature("pierna_" + key, [leg, kneeMotor]);
  feature("pantorrilla_" + key, [...[1,2,3,"4.3"].map(i => leg+"."+i), kneeMotor+".3"]);
  feature("pie_" + key, [leg+".1"]);
  feature("can_" + key, [wire+".3"]);
  feature("potencia_" + key, [wire+".7"]);
  feature("mazo_mano_" + key, [arm+".9"]);
  for (let digit=0; digit<5; digit++) {
    const root = arm+".8."+(6+digit);
    rig.groups.forEach((indices,j) => feature("dedo_"+key+"_"+digit+"_"+j, indices.map(i => root+"."+i)));
  }
}

function value(params, id) {
  const {min,max} = parameters[id];
  return Math.max(min,Math.min(max,Number(params[id]) || 0));
}
function rigid(THREE, coords) {
  return new THREE.Matrix4().compose(new THREE.Vector3(...coords.slice(0,3)),
    new THREE.Quaternion(...coords.slice(3)).normalize(), new THREE.Vector3(1,1,1));
}
function aroundX(THREE, z, degrees) {
  return new THREE.Matrix4().makeTranslation(0,0,z)
    .multiply(new THREE.Matrix4().makeRotationX(degrees*Math.PI/180))
    .multiply(new THREE.Matrix4().makeTranslation(0,0,-z));
}
function bendPoint(p, pivotZ, degrees) {
  if (!degrees) return p;
  const a=degrees*Math.PI/180, c=Math.cos(a), s=Math.sin(a), y=p.y, z=p.z-pivotZ;
  return p.set(p.x,c*y-s*z,pivotZ+s*y+c*z);
}
function closeHand(ctx, key, percent) {
  if (!percent) return;
  const {THREE,effects} = ctx;
  rig.sides[key].digits.forEach((pose,digit) => {
    const frame = rigid(THREE,pose);
    const inverse = frame.clone().invert();
    const curve = rig.moves[digit===3 ? "pulgar" : "dedos"];
    const phase = percent*(curve.length-1)/100;
    const first = Math.floor(phase);
    const blend = phase-first;
    curve[first].forEach((from,j) => {
      const to = curve[Math.min(first+1,curve.length-1)][j];
      const xyz = new THREE.Vector3(...from.slice(0,3)).lerp(new THREE.Vector3(...to.slice(0,3)),blend);
      const q = new THREE.Quaternion(...from.slice(3)).normalize()
        .slerp(new THREE.Quaternion(...to.slice(3)).normalize(),blend);
      const delta = new THREE.Matrix4().compose(xyz,q,new THREE.Vector3(1,1,1));
      const world = frame.clone().multiply(delta).multiply(inverse);
      effects.transform("dedo_"+key+"_"+digit+"_"+j,{matrix:world.transpose().toArray()});
    });
  });
}

// Los sólidos del STEP describen el mazo en reposo. En una pose se dibuja su
// línea central, interpolando entre los anclajes de torso, brazo, antebrazo y mano.
const hoses = new WeakMap();
function hoseCache(ctx) {
  const host = ctx.modelGroup;
  let cache = hoses.get(host);
  if (cache) return cache;
  cache = new Map();
  hoses.set(host,cache);
  ctx.cleanup(() => {
    for (const mesh of cache.values()) {
      mesh.removeFromParent();
      mesh.geometry.dispose();
      mesh.material.dispose();
    }
    hoses.delete(host);
  });
  return cache;
}
function routeCables(ctx,key,shoulder,elbow,wrist) {
  const {THREE,effects,modelGroup} = ctx;
  const upper = aroundX(THREE,1400,shoulder);
  const lower = upper.clone().multiply(aroundX(THREE,1100,elbow));
  const palm = lower.clone().multiply(aroundX(THREE,845,wrist));
  const cache = hoseCache(ctx);
  for (const [i,id,radius,color] of [[0,"can",1.55,0x19b4ca],[1,"potencia",1.95,0xf48137],[2,"mazo_mano",3.25,0xc5823d],
      [3,"can_motor",2.0,0x22a98e],[4,"potencia_motor",2.6,0xd67039]]) {
    const featureId = id+"_"+key;
    let mesh = cache.get(featureId);
    const flexed = i===2 ? wrist!==0 : i>=3 ? shoulder!==0 : shoulder!==0 || elbow!==0;
    if (!flexed) {
      if (mesh) mesh.visible=false;
      continue;
    }
    const points = rig.sides[key].curves[i].map(([x,y,z,u,v]) => {
      const p = new THREE.Vector3(x,y,z);
      if (i===2) return p.clone().applyMatrix4(lower).lerp(p.applyMatrix4(palm),u);
      // Each service loop bends around its pin. Interpolating endpoint positions
      // along a chord would pull the cable into the cap at a flexed elbow.
      return bendPoint(bendPoint(p,1100,elbow*v),1400,shoulder*u);
    });
    const path = new THREE.CatmullRomCurve3(points);
    const geometry = new THREE.TubeGeometry(path,points.length*2,radius,8,false);
    if (!mesh) {
      mesh=new THREE.Mesh(geometry,new THREE.MeshStandardMaterial({color,roughness:.8}));
      mesh.name=featureId;
      modelGroup.add(mesh);
      cache.set(featureId,mesh);
    } else {
      mesh.geometry.dispose();
      mesh.geometry=geometry;
      mesh.visible=true;
    }
    effects.visible(featureId,false);
  }
}
export default {
  manifest: {schemaVersion:1, step:{path:"cad/humanoide_xavier.step"},parameters,features},
  update(ctx) {
    const {params,effects} = ctx;
    effects.visible("carenados_piernas",params.carcasa_piernas !== false);
    effects.visible("carenados_brazos",params.carcasa_brazos !== false);
    function rotate(target,origin,degrees,axis=[1,0,0]) {
      if (degrees) effects.transform(target,{rotate:{axis,origin,angleDeg:degrees}});
    }
    rotate("cabeza",[0,0,1532],value(params,"cuello"),[0,0,1]);
    for (const {key,side} of sides) {
      const read = name => value(params,name+"_"+key);
      const shoulder=read("hombro"), elbow=read("codo"), wrist=read("muneca");
      const x=236*side, hipX=112*side;
      // El visor premultiplica los efectos: resolver de distal a proximal.
      closeHand(ctx,key,read("cierre"));
      rotate("mano_"+key,[x,0,845],wrist);
      rotate("antebrazo_"+key,[x,0,1100],elbow);
      rotate("brazo_"+key,[x,0,1400],shoulder);
      rotate("pie_"+key,[hipX,0,85],read("tobillo"));
      rotate("pantorrilla_"+key,[hipX,0,480],-read("rodilla"));
      rotate("pierna_"+key,[hipX,0,875],read("cadera"));
      routeCables(ctx,key,shoulder,elbow,wrist);
    }
  },
};
