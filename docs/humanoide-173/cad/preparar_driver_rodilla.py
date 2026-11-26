"""Deriva el modelo de integración del STEP de CubeMars AK54-4810-2D-A2 V3.30."""
from pathlib import Path
from build123d import Box, Color, Compound, Pos, export_step, import_step

REF=Path(__file__).resolve().parent.parent/'referencias'

def build():
    source=import_step(REF/'AK54-4810-2D-A2.step')
    pieces=[]
    for ref in source.children:
        if ref.label in ('Board','P2','P1_1'):
            # Preserve the manufacturer outline, mounting holes and connector faces.
            # Keep solids as a single occurrence to avoid 3000 internal mesh nodes.
            item=Compound(ref.solids())
            paint={'Board':'#367556','P2':'#C59A41','P1_1':'#DEDDD0'}[ref.label]
        else:
            if not ref.label.startswith(('U','Q','L','C')): continue
            b=ref.bounding_box()
            if max(b.size)<3 or min(b.size)<.15: continue
            item=Pos(*b.center())*Box(*b.size)
            paint='#343C41' if ref.label.startswith(('U','Q')) else '#AD976F'
        item.label,item.color=ref.label,Color(paint)
        pieces.append(item)
    board=Compound(label='AK54-4810-2D-A2 - integracion V3.30',children=pieces)
    export_step(board, REF/'AK54-4810-2D-A2-integracion.step')

if __name__=='__main__':
    build()
