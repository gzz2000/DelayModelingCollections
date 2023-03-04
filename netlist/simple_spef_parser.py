# a deadly simple spef parser in python.
# only expects and parses one net.

class Parasitics(object):
    def _generate_names(self):
        for endpoint, _ in self.endpoints_name:
            yield endpoint
        for cap in self.caps_name:
            if len(cap) == 2:
                yield cap[0]
            else:
                yield cap[0]
                yield cap[1]
        for res in self.ress_name:
            yield res[0]
            yield res[1]
        
    def __init__(self, endpoints, caps, ress):
        self.endpoints_name = endpoints
        self.caps_name = caps
        self.ress_name = ress

        self.name2id = {}
        self.names = []
        for name in self._generate_names():
            if name not in self.name2id:
                self.name2id[name] = len(self.name2id)
                self.names.append(name)

        self.endpoints = [
            (self.name2id[name], io)
            for name, io in endpoints
        ]
        self.grounded_caps = [
            (self.name2id[cap[0]], cap[1])
            for cap in caps if len(cap) == 2
        ]
        self.coupling_caps = [
            (self.name2id[cap[0]], self.name2id[cap[1]], cap[2])
            for cap in caps if len(cap) == 3
        ]
        self.ress = [
            (self.name2id[res[0]], self.name2id[res[1]], res[2])
            for res in ress
        ]
        self.n = len(self.name2id)

        self.sink_cell_caps = []

    def __repr__(self):
        return (f'<Parasitics object at {hex(id(self))}: '
                f'{self.n} nodes, '
                f'{len(self.grounded_caps)} grounded / '
                f'{len(self.coupling_caps)} coupling caps, '
                f'{len(self.ress)} ress>')

def parse_file(path):
    with open(path) as f:
        endpoints = []
        caps = []
        ress = []
        reading = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            elif line.startswith('*D_NET'):
                reading = None
                continue
            elif line.startswith('*END'):
                reading = None
                break
            elif line.startswith('*I'):
                pin, io = line.split()[1:]
                endpoints.append((pin, io))
            elif line.startswith('*CONN'):
                continue
            elif line.startswith('*CAP'):
                reading = 'CAP'
            elif line.startswith('*RES'):
                reading = 'RES'
            else:
                c = line.split()[1:]
                c[-1] = float(c[-1])
                if reading == 'CAP': caps.append(tuple(c))
                elif reading == 'RES': ress.append(tuple(c))
                else: assert False, 'bad reading state'
    return Parasitics(endpoints=endpoints, caps=caps, ress=ress)
