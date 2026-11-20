"""Parcel geometry in metres, shared by USD authoring and future navigation tasks."""
from dataclasses import dataclass
from math import atan, cos, degrees, exp, pi, sin, tanh


@dataclass(frozen=True)
class Field:
    name: str
    row_spacing: float
    plant_spacing: float
    plants_per_row: int
    bend: float
    grade: float
    crossfall: float
    rut_depth: float
    bed_height: float
    bed_width: float
    seed: int
    length: float = 24.0
    headland: float = 5.0

    @property
    def fence_y(self):
        return self.row_spacing + self.bend + 2.3

    def bend_y(self, x):
        return self.bend * (.72 * sin(.23*x - .6) + .28 * sin(.47*x + .3))

    def row_y(self, x, row):
        return self.bend_y(x) + row*self.row_spacing

    def row_yaw(self, x):
        slope = self.bend * (.72*.23*cos(.23*x-.6) + .28*.47*cos(.47*x+.3))
        return degrees(atan(slope))

    def plant_x(self, n):
        return (self.length - (self.plants_per_row-1)*self.plant_spacing)/2 + n*self.plant_spacing

    def bed_and_tracks(self, x, y):
        lateral = y-self.bend_y(x)
        # The first and last metre taper into open turning areas.
        planted = .5*(tanh(1.9*x) - tanh(1.9*(x-self.length)))
        d = min(abs(lateral-i*self.row_spacing) for i in (-1, 0, 1))
        bed = exp(-(d/(self.bed_width*.5))**4) * planted
        lane = abs(abs(lateral)-self.row_spacing*.5)
        track = exp(-((lane-.51)/.145)**2) * planted
        return bed, track

    def height(self, x, y):
        bed, track = self.bed_and_tracks(x, y)
        waves = .042*sin(.45*x + .31*y) + .018*sin(.91*x - .77*y)
        return (self.grade*x + self.crossfall*y + waves + self.bed_height*bed
                - self.rut_depth*track*(.80 + .20*cos(.72*x)))

    def reference_route(self):
        """Centreline waypoints, metres, outbound through the headland and back."""
        start = -self.headland
        steps = round((self.length-start)/.5)
        xs = [start+(self.length-start)*i/steps for i in range(steps+1)]
        xy = [(x,self.row_y(x,-.5)) for x in xs]
        radius = self.row_spacing/2
        for i in range(1,25):
            angle = pi*i/24
            x = self.length+radius*sin(angle)
            xy.append((x,self.bend_y(x)-radius*cos(angle)))
        xy.extend((x,self.row_y(x,.5)) for x in reversed(xs[:-1]))
        return [(x,y,self.height(x,y)) for x,y in xy]


FIELDS = {
    "citrus": Field("naranjal", 6., 3., 8, 1.25, .035, .020, .045, .12, 1.45, 731),
    "banana": Field("platanal", 4., 3., 8, 1.00, .024, -.021, .080, .15, 1.25, 739),
    "vineyard": Field("vinedo", 3., 1.8, 13, .48, .030, .027, .042, .10, .88, 743),
    "berries": Field("arandanal", 3., 1., 24, .75, .021, -.020, .060, .35, 1., 751),
}
