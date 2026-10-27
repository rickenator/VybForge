import struct, random, math
def bits(x): return struct.unpack('<q',struct.pack('<d',x))[0]
def fb(b): return struct.unpack('<d',struct.pack('<q',((b+2**63)%2**64)-2**63))[0]
def vs(x,n):
    if x<=0: return 0.0
    y=x
    for _ in range(n): y=0.5*(y+x/y)
    return y
def seed(x,n=6):
    y=x*fb(6910469410427058089 - int(bits(x)/2))
    for _ in range(n): y=0.5*(y+x/y)
    return y
random.seed(7); bad=0; N=300000
for i in range(N):
    x=10**random.uniform(-9,9)
    if random.random()<0.1: x=float(random.randint(1,10**6))/random.choice([1,7,1024])
    if vs(x,40)!=seed(x): bad+=1
    if bad and bad<4 and vs(x,40)!=seed(x): print("diff",repr(x),vs(x,40),seed(x))
print("VSQRT40_SEED6 mismatches",bad,"of",N)
