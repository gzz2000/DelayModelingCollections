# simple elmore delay
# dynamic programming on tree

def calc_elmore(rc):
    assert len(rc.coupling_caps) == 0, \
        'Elmore delay don\'t support coupling caps'
    assert len(rc.grounded_caps) <= rc.n and len(rc.ress) == rc.n - 1, \
        'Must be a tree'
    caps = [0. for i in range(rc.n)]
    load = [0. for i in range(rc.n)]
    delay = [0. for i in range(rc.n)]
    ldelay = [0. for i in range(rc.n)]
    beta = [0. for i in range(rc.n)]
    sibling = [[] for i in range(rc.n)]
    for a, c in rc.grounded_caps:
        caps[a] = c
    for a, b, r in rc.ress:
        sibling[a].append((b, r))
        sibling[b].append((a, r))
    
    def dfs_load(u, f):
        load[u] = caps[u]
        for v, r in sibling[u]:
            if v == f: continue
            dfs_load(v, u)
            load[u] += load[v]
            
    def dfs_delay(u, f):
        for v, r in sibling[u]:
            if v == f: continue
            delay[v] = delay[u] + r * load[v]
            dfs_delay(v, u)

    def dfs_ldelay(u, f):
        ldelay[u] = caps[u] * delay[u]
        for v, r in sibling[u]:
            if v == f: continue
            dfs_ldelay(v, u)
            ldelay[u] += ldelay[v]
            
    def dfs_beta(u, f):
        for v, r in sibling[u]:
            if v == f: continue
            beta[v] = beta[u] + r * ldelay[v]
            dfs_beta(v, u)

    dfs_load(0, None)
    dfs_delay(0, None)
    dfs_ldelay(0, None)
    dfs_beta(0, None)

    return delay, beta

if __name__ == '__main__':
    from netlist import tree_rc
    delay, beta = calc_elmore(tree_rc)
    for eid, io in tree_rc.endpoints:
        print(f'Endpoint {tree_rc.names[eid]} ({io}) index {eid}: ', end='')
        print(f'Delay = {delay[eid]:.05}, Beta = {beta[eid]:.05}')
