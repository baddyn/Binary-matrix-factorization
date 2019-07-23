from PIL import Image
from numpy import *
im=Image.open("update.png")
a=matrix(im)
u=a[:,0:3]
v=a[:,3:6]
x=u*v.transpose()
xi,xj=x.shape
for i in range(xi):
  for j in range(xj):
    if x[i,j]>=1:
      x[i,j]=255
ai,aj=a.shape
for i in range(ai):
  j=6
  while j<aj and a[i,j]!=255:
    x[i,a[i,j]]=((x[i,a[i,j]]+1)%256)*255
    j+=1
xi,xj=x.shape
x=x.astype(uint8)
im=Image.fromarray(x)
im.save("qr.png")
