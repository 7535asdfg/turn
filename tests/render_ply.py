import numpy as np, sys
from PIL import Image
path=sys.argv[1] if len(sys.argv)>1 else 'tests/synthetic/recon.ply'
# parse ascii ply
verts=[];colors=[]
with open(path) as f:
    header=True
    nv=0
    for line in f:
        if header:
            if line.startswith('element vertex'): nv=int(line.split()[-1])
            if line.strip()=='end_header': header=False
            continue
        p=line.split()
        if len(p)<3: continue
        verts.append([float(p[0]),float(p[1]),float(p[2])])
        colors.append([int(p[3]),int(p[4]),int(p[5])])
        if len(verts)>=nv: break
V=np.array(verts);C=np.array(colors,dtype=float)/255.0
print('verts',V.shape)
# center and normalize
c=(V.min(0)+V.max(0))/2
V=V-c
scale=np.max(np.abs(V))
V=V/scale
# render 3 views
for ai,ang in enumerate([0,45,90,180]):
    a=np.radians(ang)
    Rz=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
    Vr=V@Rz.T
    # camera looks along -Z from z=3, x right, y up
    f=1.5; W=512;H=512
    z=Vr[:,2]
    x=(Vr[:,0]*f/(3.0-z))*W*0.5+W/2
    y=(-Vr[:,1]*f/(3.0-z))*H*0.5+H/2
    img=np.zeros((H,W,3),dtype=np.float32)
    zb=np.full((H,W),np.inf)
    order=np.argsort(z)[::-1]
    for i in order:
        xi=int(x[i]);yi=int(y[i])
        if 0<=xi<W and 0<=yi<H and z[i]<zb[yi,xi]:
            zb[yi,xi]=z[i];img[yi,xi]=C[i]
    # dilate a bit
    Image.fromarray((np.clip(img,0,1)*255).astype('uint8')).save(f'tests/synthetic/recon_{ang}.png')
    print('saved',ai,ang)
