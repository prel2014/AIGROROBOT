"""Arrange actual simulator captures into a labelled gallery; generates no scene imagery."""
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from dexman.tasks.manipulation.scenarios import SITUATIONS


def main():
    folder = ROOT/"assets/previews"
    font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf",18)
    gallery = Image.new("RGB",(960,4*400),(24,28,33))
    draw = ImageDraw.Draw(gallery)
    for i,s in enumerate(SITUATIONS):
        x,y = (i%2)*480,(i//2)*400
        source = Image.open(folder/s.name/"overview.png").convert("RGB")
        gallery.paste(source.resize((480,360)),(x,y+40))
        draw.text((x+10,y+9),f"{i+1}. {s.name} · {s.label}",font=font,fill="white")
    gallery.save(folder/"environments.png")
    panel = Image.new("RGB",(896,264),(24,28,33))
    draw = ImageDraw.Draw(panel)
    for i,(name,label) in enumerate((("camera","RGB frontal"),("camera_depth","Profundidad frontal"),
                                    ("camera_secondary","RGB opuesto"),("camera_secondary_depth","Profundidad opuesta"))):
        panel.paste(Image.open(folder/"clear_bench"/(name+".png")).convert("RGB"),(i*224,40))
        draw.text((i*224+7,10),label,font=font,fill="white")
    panel.save(folder/"dual_rgbd.png")
    print("Created gallery from eight real simulator captures and both RGB-D views")


if __name__ == "__main__":
    main()
