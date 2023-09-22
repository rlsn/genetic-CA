def color_mapping(genome):
    n = [0,0,0]
    rgb = [0,0,0]
    for i,g in enumerate(genome):
        rgb[i%3]+=g//2**24
        n[i%3]+=1
    return tuple([max(min(c//m,255),0) for c,m in zip(rgb,n)])
