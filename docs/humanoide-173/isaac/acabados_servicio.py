"""Mapas PBR periódicos y materiales ópticos de los ocho entornos de servicio."""
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
from pxr import Sdf, UsdGeom, UsdShade

from geometria_escenarios import REPETICION, Taller, mapa


def ruido(rng, u, v, nx, ny):
    """Interpolación periódica; la frecuencia de cada eje se expresa en ciclos por baldosa."""
    nodos = rng.uniform(-1, 1, (ny, nx)).astype("f")
    x, y = u * nx, v * ny
    i, j = x.astype(int), y.astype(int)
    a, b = x - i, y - j
    a, b = a * a * (3 - 2 * a), b * b * (3 - 2 * b)
    abajo = (1 - a) * nodos[j % ny, i % nx] + a * nodos[j % ny, (i + 1) % nx]
    arriba = (1 - a) * nodos[(j + 1) % ny, i % nx] + a * nodos[(j + 1) % ny, (i + 1) % nx]
    return abajo * (1 - b) + arriba * b


def guardar(carpeta, nombre, color, altura, rugosidad, fuerza):
    dx = (np.roll(altura, -1, 1) - np.roll(altura, 1, 1)) * fuerza
    dy = (np.roll(altura, -1, 0) - np.roll(altura, 1, 0)) * fuerza
    normal = np.stack((-dx, dy, np.ones_like(altura)), axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    for canal, datos in (("color", color), ("normal", .5 + .5 * normal), ("rugosidad", rugosidad)):
        img = Image.fromarray(np.uint8(np.clip(datos, 0, 1) * 255 + .5))
        img.save(carpeta / f"{nombre}_{canal}.png")


def matiz(base, valor):
    return np.array(base) * valor[..., None]


@lru_cache(maxsize=1)
def generar_texturas(carpeta):
    carpeta.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(601)
    v, u = (np.indices((768, 768), dtype="f") + .5) / 768
    fino = rng.uniform(-1, 1, u.shape).astype("f")
    nube = ruido(rng, u, v, 8, 8)
    grano = ruido(rng, u, v, 110, 110)
    poros = np.clip((-grano - .47) * 3, 0, 1)
    hormigon = .88 + .12 * nube + .09 * grano + .055 * fino - .20 * poros
    guardar(carpeta, "cemento", matiz((.67, .65, .62), hormigon), grano * .22 - poros * .6,
            .75 + .09 * nube + .06 * poros, 1.15)

    # Áridos de contorno irregular, pigmentos diferentes y juntas perimetrales.
    ix, iy = (u * 53).astype(int), (v * 53).astype(int)
    ox, oy = rng.uniform(.27, .73, (2, 53, 53))
    px = (u * 53) % 1 - ox[iy, ix]
    py = (v * 53) % 1 - oy[iy, ix]
    faceta = np.abs(px + py * .42) + np.abs(py - px * .16) * 1.12
    chip = faceta < rng.uniform(.14, .45, (53, 53))[iy, ix] + grano * .07
    pigmentos = np.array([[.83, .78, .65], [.39, .40, .36], [.66, .48, .37], [.89, .88, .82]])
    cemento = matiz((.68, .65, .60), 1 + .035 * nube + .028 * fino)
    terrazo = np.where(chip[..., None], pigmentos[rng.integers(0, 4, (53, 53))[iy, ix]], cemento)
    junta = np.minimum.reduce([u, v, 1 - u, 1 - v]) < .006
    guardar(carpeta, "terrazo", np.where(junta[..., None], (.42, .41, .38), terrazo),
            chip * .045 - junta * .82 + fino * .015, .205 + .075 * chip + .37 * junta + .022 * nube, 1.20)

    cepillo = ruido(rng, u, v, 3, 310) * .85 + fino * .15
    guardar(carpeta, "cepillado", matiz((.92, .93, .94), .97 + .025 * cepillo), .16 * cepillo,
            .20 + .09 * cepillo + .016 * nube, .9)
    fibra = ruido(rng, u, v, 170, 44) * .55 + fino * .45
    ondulado = np.cos(32 * 2 * np.pi * u)
    guardar(carpeta, "kraft", matiz((.69, .51, .32), 1 + .13 * fibra + .048 * ondulado + .09 * nube),
            .28 * fibra + .18 * ondulado, .82 + .068 * fibra, 1.05)
    urdimbre = np.cos(96 * 2 * np.pi * u)
    trama = np.cos(88 * 2 * np.pi * v)
    tejido = (urdimbre + trama) * .28 + .40 * urdimbre * trama + .08 * fino
    guardar(carpeta, "lona", matiz((.96, .96, .94), .90 + .09 * tejido + .045 * nube), tejido,
            .72 + .10 * tejido, 1.15)
    junta_az = np.minimum.reduce([u, v, 1 - u, 1 - v]) < .013
    guardar(carpeta, "azulejo",
            np.where(junta_az[..., None], (.51, .50, .46), matiz((.85, .89, .85), 1 + .018 * nube)),
            .03 * nube - junta_az * .65, .12 + .025 * nube + .51 * junta_az, 1.30)
    estria = np.maximum(0, np.cos((u + v) * np.pi * 2 * 14))**3
    guardar(carpeta, "goma", matiz((.19, .20, .20), .91 + .12 * estria + .11 * fino),
            estria * .62 + .07 * grano, .71 + .10 * grano, 1.2)
    guardar(carpeta, "piel", matiz((1., .99, .95), .94 + .04 * nube + .02 * fino - .04 * poros),
            .06 * grano - .12 * poros, .32 + .07 * nube + .06 * poros, .42)
    lat = (.5 - v) * np.pi
    xpan = np.cos(2 * np.pi * u) * np.cos(lat)
    ypan = np.sin(2 * np.pi * u) * np.cos(lat)
    cortes = np.zeros_like(v)
    for n in (-1, 0, 1):
        distancia = ((xpan - .44 * n - .27 * ypan) / .11)**2 + (ypan / .60)**2
        cortes = np.maximum(cortes, np.clip((1 - distancia + .13 * grano) * 3, 0, 1)) * (lat > 0)
    tostado = .83 + .13 * nube + .07 * grano + .035 * fino - .10 * poros
    pan = matiz((.78, .53, .285), tostado)
    miga = matiz((.91, .78, .56), .94 + .08 * grano - .13 * poros)
    guardar(carpeta, "pan",
            pan * (1 - cortes[..., None]) + miga * cortes[..., None],
            -.29 * poros + .09 * grano - .32 * cortes, .78 + .05 * poros + .04 * nube, .68)
    nervio = np.exp(-abs(u - .5) * 110)
    venas = np.maximum(0, np.cos(2 * np.pi * (11 * v - 4 * abs(u - .5))))**12
    hoja = .79 + .16 * nube + .12 * venas + .18 * nervio
    guardar(carpeta, "hoja", matiz((.92, 1., .80), hoja), .29 * nervio + .12 * venas,
            .36 + .07 * nube - .06 * nervio, 1.3)
    hilera = np.floor(v * 25)
    escama = ((u * 46 + .5 * (hilera % 2)) % 1 - .5)**2 * 2.4
    borde = np.exp(-((v * 25 % 1 - .23 - escama) / .095)**2)
    dorso = np.clip(.50 + .7 * np.sin(2 * np.pi * u) * np.cos(lat), 0, 1)
    pez = (1 - dorso[..., None]) * np.array([.87, .86, .79]) + dorso[..., None] * np.array([.20, .33, .36])
    brillo = .91 + .06 * nube - .10 * borde
    guardar(carpeta, "escamas", pez * brillo[..., None], .13 * borde + .025 * grano,
            .24 + .10 * borde + .02 * nube, .72)
    # Engobe, poro abierto y estrías horizontales del torneado.
    torno = np.cos(2 * np.pi * (v * 49 + .13 * ruido(rng, u, v, 4, 6)))
    sal = np.clip((ruido(rng, u, v, 24, 33) + .7 * nube - .65) * 1.8, 0, 1)
    barro = matiz((.67, .35, .215), .92 + .11 * nube + .06 * torno + .08 * fino - .11 * poros)
    barro += matiz((.25, .24, .21), sal)
    guardar(carpeta, "arcilla", barro, .10 * torno - .33 * poros + .06 * grano, .74 + .07 * nube, 1.8)
    turba = matiz((.30, .20, .115), .88 + .22 * nube + .38 * grano + .20 * fino)
    guardar(carpeta, "turba", turba, .48 * grano + .12 * fino, .88 + .045 * grano, 1.7)
    # Moqueta de transporte de bucle corto con motivos escalonados de tres colores.
    fila = np.floor(v * 8)
    escalon = (np.floor(u * 12) + 2 * fila) % 9
    motivo = (escalon == 0) & ((v * 8) % 1 < .65)
    segundo = (escalon == 1) & ((v * 8) % 1 > .64)
    tapiz = np.where(motivo[..., None], (.23, .62, .61),
                     np.where(segundo[..., None], (.82, .58, .24), (.14, .32, .38)))
    guardar(carpeta, "tapiceria",
            tapiz * (.91 + .12 * tejido + .04 * fino)[..., None], .51 * tejido + .13 * fino,
            .83 + .06 * tejido, 1.24)
    citrico = ruido(rng, u, v, 145, 95)
    glandula = np.clip((citrico - .30) * 2.7, 0, 1)
    guardar(carpeta, "citrico", matiz((1., .98, .90), .91 + .065 * nube - .16 * glandula + .045 * fino),
            .10 * citrico - .32 * glandula, .41 + .08 * glandula + .03 * nube, .72)
    x, y = 2 * u - 1, 2 * v - 1
    r = np.hypot(x, y)
    a = np.arctan2(y, x)
    fase = 10 * a + .16 * np.sin(3 * a) + .21 * np.sin(a + r * 3)
    tabique = np.abs(np.sin(fase * .5)) * r
    membrana = (tabique < .035) | (r < .087 + .018 * np.cos(5 * a)) | (r > .87 + .009 * nube)
    sacos = ruido(rng, u, v, 110, 145)
    pulpa = matiz((1., .64, .125), .86 + .14 * sacos + .085 * np.cos(fase) + .05 * nube)
    corte = np.where(membrana[..., None], matiz((.99, .89, .65), .97 + .055 * grano), pulpa)
    corte = np.where((r > .946)[..., None], matiz((.99, .48, .056), .95 + .05 * grano), corte)
    guardar(carpeta, "pulpa", corte, .15 * membrana + .15 * sacos, .20 + .19 * membrana + .065 * sacos,
            1.35)


TEXTURAS = {
    "terrazo": ("terrazo", 0.),
    "hormigon": ("cemento", 0.),
    "aluminio": ("cepillado", .96),
    "inox": ("cepillado", 1.),
    "carton": ("kraft", 0.),
    "tejido": ("lona", 0.),
    "azulejo": ("azulejo", 0.),
    "piso_goma": ("goma", 0.),
    "corteza": ("pan", 0.),
    "escamas": ("escamas", .28),
    "hoja_cultivo": ("hoja", 0.),
    "hoja_tierna": ("hoja", 0.),
    "barro": ("arcilla", 0.),
    "barro_ocre": ("arcilla", 0.),
    "tierra_maceta": ("turba", 0.),
    "pulpa_naranja": ("pulpa", 0.),
    "asiento_bus": ("tapiceria", 0.),
    "naranja": ("citrico", 0.),
    "limon": ("citrico", 0.),
    **{
        nombre: ("piel", 0.)
        for nombre in ("tomate", "berenjena")
    },
}


class TallerServicio(Taller):
    repeticion = {
        **REPETICION, "aluminio": (.48, .48),
        "inox": (.45, .45),
        "carton": (.32, .32),
        "azulejo": (.20, .20),
        "piso_goma": (.32, .32),
        "asiento_bus": (.27, .27)
    }

    def acabado(self, prim, nombre):
        # El metal mecanizado de tubos y tornillos conserva un brillo uniforme.
        if nombre in ("aluminio", "inox") and not prim.IsA(UsdGeom.Mesh):
            nombre = "metal_torneado"
        super().acabado(prim, nombre)


def preparar_materiales(t):
    for nombre, mat in t.materiales.items():
        s = UsdShade.Shader.Get(t.stage, mat.GetPath().AppendChild("Superficie"))
        if nombre in ("vidrio", "vidrio_satinado"):
            pulido = nombre == "vidrio"
            rugosidad = .035 if pulido else .29
            s.GetInput("diffuseColor").Set((.91, .97, .95))
            s.GetInput("roughness").Set(rugosidad)
            s.GetInput("metallic").Set(0.)
            s.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(.11 if pulido else .27)
            s.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.52)
            mdl = UsdShade.Shader.Define(t.stage, mat.GetPath().AppendChild("Transmision"))
            mdl.SetSourceAsset(Sdf.AssetPath("./materiales/vidrio_arquitectonico.mdl"), "mdl")
            mdl.SetSourceAssetSubIdentifier("vidrio_arquitectonico", "mdl")
            mdl.CreateInput("rugosidad", Sdf.ValueTypeNames.Float).Set(rugosidad)
            mat.CreateSurfaceOutput("mdl").ConnectToSource(mdl.CreateOutput(
                "out", Sdf.ValueTypeNames.Token))
        elif nombre in ("espejo", "metal_torneado", "laton", "ceramica", "laca_bus"):
            color, metal, rugosidad = {
                "espejo": ((.97, .97, .97), 1., .012),
                "metal_torneado": ((.84, .87, .89), .98, .20),
                "laton": ((.83, .54, .22), .94, .21),
                "ceramica": ((.88, .85, .78), 0., .135),
                "laca_bus": ((.03, .20, .21), .20, .24)
            }[nombre]
            s.GetInput("diffuseColor").Set(color)
            s.GetInput("metallic").Set(metal)
            s.GetInput("roughness").Set(rugosidad)
            if nombre in ("ceramica", "laca_bus"):
                s.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(.78)
                s.CreateInput("clearcoatRoughness", Sdf.ValueTypeNames.Float).Set(.09)
        elif nombre in TEXTURAS:
            juego, metal = TEXTURAS[nombre]
            lector = UsdShade.Shader.Define(t.stage, mat.GetPath().AppendChild("Coordenadas"))
            lector.CreateIdAttr("UsdPrimvarReader_float2")
            lector.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            lector.CreateOutput("result", Sdf.ValueTypeNames.Float2)
            base = f"servicio_pbr/{juego}_"
            color = mapa(t.stage, mat, "Albedo", base + "color.png", lector, "sRGB")
            if juego in ("piel", "citrico") or nombre == "tejido":
                tinte = s.GetInput("diffuseColor").Get()
                color.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set((*tinte, 1.))
            if nombre == "barro_ocre":
                color.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set((1.46, 2.14, .85, 1.))
            if juego == "hoja":
                tinte = (.16, .34, .08) if nombre == "hoja_cultivo" else (.30, .52, .14)
                color.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set((*tinte, 1.))
            s.GetInput("diffuseColor").ConnectToSource(color.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
            rough = mapa(t.stage, mat, "Rugosidad", base + "rugosidad.png", lector, "raw")
            s.GetInput("roughness").ConnectToSource(rough.CreateOutput("r", Sdf.ValueTypeNames.Float))
            bump = mapa(t.stage, mat, "Normal", base + "normal.png", lector, "raw")
            bump.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set((2., 2., 2., 1.))
            bump.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set((-1., -1., -1., 0.))
            s.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
                bump.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
            s.GetInput("metallic").Set(metal)
