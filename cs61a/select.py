def select(p,i):
    return p(i)

def pair(x,y):
    def get(index):
        if index==0:
            return x
        elif index==1:
            return y
    return get