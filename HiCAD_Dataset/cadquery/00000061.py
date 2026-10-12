import cadquery as cq
w0=cq.Workplane('XY',origin=(-100,-100,-5))
r=w0.sketch().face(w0.sketch().segment((0,0),(200,0)).segment((200,0),(200,200)).segment((200,200),(20,200)).arc((20,200),(14,186),(0,180)).segment((0,180),(0,0)).assemble()).finalize().extrude(10)