from nimfa import *
from numpy import *
from PIL import *
import math
  
def funcnmf(a,r):
  nmf=Nmf(a,rank=r)
  s=nmf()
  u=s.basis()
  v=s.coef()
  y,x=u.shape
  U=matrix(zeros((y,x),dtype=float))
  for j in range(x):
    r=math.sqrt(max(v[j,:].transpose())/max(u[:,j]))
    for i in range(y):
      U[i,j]=u[i,j]*r
  x,y=v.shape
  V=matrix(zeros((x,y),dtype=float))
  for i in range(x):
    r=math.sqrt(max(u[:,i])/max(v[i,:].transpose()))
    for j in range(y):
      V[i,j]=v[i,j]*r
  return (U,V)


def update(a,u,v,l):
  x,y=u.shape
  U=matrix(empty((x,y),dtype=float))
  xv,yv=v.shape
  V=matrix(empty((xv,yv),dtype=float))
  ut=u.transpose()
  wtx=ut*a
  wtwh=ut*u*v
  for i in range(xv):
    for j in range(yv):
      V[i,j]=v[i,j]*((wtx[i,j]+3*l*(v[i,j]**2))/(wtwh[i,j]+2*l*(v[i,j]**3)+l*v[i,j]))
  vt=v.transpose()
  htx=a*vt
  hthw=u*v*vt
  for i in range(x):
    for j in range(y):
      U[i,j]=u[i,j]*((htx[i,j]+3*l*(u[i,j]**2))/(hthw[i,j]+2*l*(u[i,j]**3)+l*u[i,j]))
  return (U,V)


def roundoff(u):
  a,b=u.shape
  for i in range(a):
    for j in range(b):
      if u[i,j]>0.5:
        u[i,j]=1
      else:
        u[i,j]=0

for t1 in range(8):
  for t2 in range(1,7):
    filename="L\m_"+str(t1)+"\\1_"+str(t2)+"_L_"+str(t1)+".png"
    fil2="rank2\L__m_"+str(t1)+"__1_"+str(t2)+"_L_"+str(t1)+".png"
    jpgfile1=Image.open(filename,mode='r')
    x=matrix(jpgfile1)
    a,b=x.shape
    for i in range(a):
      for j in range(b):
        if x[i,j]>1:
          x[i,j]=1

    r=2
    u,v=funcnmf(x,r)
    steps=5000
    Lam=1
    for i in range(steps):
      u,v=update(x,u,v,Lam)
    roundoff(u)
    roundoff(v)
    c=0
    ai,bi=x.shape
    uv=u*v
    m=0
    cur=0
    for i in range(ai):
      cur=0
      for j in range(bi):
        if uv[i,j]>1:
          uv[i,j]=1
        if x[i,j]!=uv[i,j]:
          c+=1
          cur+=1
      if cur>m:
        m=cur
    res=matrix(ones((ai,2*r+m),dtype=int))
    res=res*255
    for i in range(ai):
      for j in range(r):
        res[i,j]=u[i,j]*255
        res[i,j+r]=v[j,i]*255
    for i in range(ai):
      p=2*r
      for j in range(bi):
        if x[i,j]!=uv[i,j]:
          res[i,p]=j
          p+=1

    res=res.astype(uint8)
    im=Image.fromarray(res)
    im.save(fil2)
    print((c/x.size)*100)


