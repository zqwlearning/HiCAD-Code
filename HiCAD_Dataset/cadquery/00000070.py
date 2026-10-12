import cadquery as cq
w0=cq.Workplane('ZX',origin=(-32,9,-75))
w1=cq.Workplane('ZX',origin=(26,9,74))
w2=cq.Workplane('ZX',origin=(-39,9,-47))
r=w0.sketch().face(w0.sketch().arc((0,0),(-22,43),(22,61)).segment((22,61),(131,7)).arc((131,7),(168,22),(150,58)).segment((150,58),(151,63)).arc((151,63),(173,20),(129,2)).segment((129,2),(20,56)).arc((20,56),(-17,41),(1,5)).segment((1,5),(0,0)).assemble()).finalize().extrude(-18).union(w1.sketch().face(w1.sketch().segment((0,0),(-28,7)).segment((-28,7),(-27,13)).segment((-27,13),(1,5)).segment((1,5),(0,0)).assemble()).finalize().extrude(-18)).union(w2.sketch().face(w2.sketch().segment((0,0),(-28,7)).segment((-28,7),(-27,13)).segment((-27,13),(1,5)).segment((1,5),(0,0)).assemble()).finalize().extrude(-18))