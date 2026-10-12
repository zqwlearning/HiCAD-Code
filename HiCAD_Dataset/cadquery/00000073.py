import cadquery as cq
w0=cq.Workplane('XY',origin=(-100,-54,43))
w1=cq.Workplane('XY',origin=(95,-54,43))
w2=cq.Workplane('XY',origin=(-100,29,43))
r=w0.sketch().face(w0.sketch().segment((0,0),(5,0)).segment((5,0),(5,12)).segment((5,12),(0,12)).segment((0,12),(0,0)).assemble()).finalize().extrude(-86).union(w1.sketch().face(w1.sketch().segment((0,0),(5,0)).segment((5,0),(5,12)).segment((5,12),(0,12)).segment((0,12),(0,0)).assemble()).finalize().extrude(-86)).union(w2.sketch().face(w2.sketch().segment((0,0),(200,0)).segment((200,0),(200,25)).segment((200,25),(0,25)).segment((0,25),(0,0)).assemble()).finalize().extrude(-86))