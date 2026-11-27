const parameters = {};
const features = {};
const ejes = [
  ["giro", "Giro del muslo", -20, 20],
  ["apertura", "Aducción (−) / abducción (+)", -8, 18],
  ["flexion", "Flexión", -15, 35]
];
for (const [indice, lado] of ["izquierda", "derecha"].entries()) {
  ejes.forEach(([eje, titulo, min, max], j) => {
    const id = eje + "_" + lado;
    parameters[id] = {type:"number", label:titulo + " · " + lado, min, max, default:0, step:1, unit:"°"};
    features[id] = {ref:"#o1." + (2 + 3 * indice + j)};
  });
}
export default {
  manifest: {schemaVersion:1, step:{path:"cad/caderas_tres_ejes.step"}, parameters, features},
  update({THREE, params, effects}) {
    for (const [lado, signo] of [["izquierda",-1],["derecha",1]]) {
      const x = signo * 112, z = 875;
      const matriz = new THREE.Matrix4().makeTranslation(x,0,z);
      for (const [i,[eje,,min,max]] of ejes.entries()) {
        const id = eje + "_" + lado;
        const rad = Math.max(min,Math.min(max,Number(params[id]) || 0)) * Math.PI / 180;
        const rot = new THREE.Matrix4();
        if (i === 0) rot.makeRotationZ(-signo * rad);
        if (i === 1) rot.makeRotationY(-signo * rad);
        if (i === 2) rot.makeRotationX(rad);
        matriz.multiply(rot);
        const delta = matriz.clone().multiply(new THREE.Matrix4().makeTranslation(-x,0,-z));
        effects.transform(id, {matrix:delta.transpose().toArray()});
      }
    }
  }
};
