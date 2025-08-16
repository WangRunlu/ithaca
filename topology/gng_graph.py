import numpy as np

class GNG:
    def __init__(self, eps_b=0.05, eps_n=0.005, age_max=40, insert_every=100, max_nodes=800):
        self.eps_b, self.eps_n, self.age_max = eps_b, eps_n, age_max
        self.insert_every, self.max_nodes = insert_every, max_nodes
        self.nodes, self.errors, self.adj, self.t = [], [], {}, 0

    def _add_node(self, x, y):
        self.nodes.append([x, y]); self.errors.append(0.0); self.adj[len(self.nodes)-1] = {}

    def _add_edge(self, i, j):
        self.adj[i][j] = 0; self.adj[j][i] = 0

    def _remove_old_edges(self):
        rm = []
        for i in list(self.adj.keys()):
            for j, age in list(self.adj[i].items()):
                if age > self.age_max: rm.append((i,j))
        for i,j in rm:
            self.adj[i].pop(j, None); self.adj[j].pop(i, None)

    def _nearest_two(self, p):
        if len(self.nodes) < 2: return None, None
        arr = np.array(self.nodes); d2 = np.sum((arr - p)**2, axis=1)
        b = int(np.argmin(d2)); d2[b] = np.inf; n = int(np.argmin(d2)); return b, n

    def fit(self, X):
        if len(self.nodes) == 0:
            i = np.random.randint(0, len(X)); j = (i + len(X)//2) % len(X)
            self._add_node(X[i,0], X[i,1]); self._add_node(X[j,0], X[j,1]); self._add_edge(0,1)
        for p in X:
            self.t += 1; p = np.asarray(p)
            b, n = self._nearest_two(p)
            if b is None: continue
            for j in list(self.adj[b].keys()): self.adj[b][j] += 1
            pb = np.array(self.nodes[b]); self.nodes[b] = list(pb + self.eps_b * (p - pb))
            for j in self.adj[b].keys():
                pj = np.array(self.nodes[j]); self.nodes[j] = list(pj + self.eps_n * (p - pj))
            self._add_edge(b, n)
            self.errors[b] += float(np.sum((p - pb)**2))
            self._remove_old_edges()
            if self.t % self.insert_every == 0 and len(self.nodes) < self.max_nodes:
                q = int(np.argmax(self.errors))
                if len(self.adj[q]) == 0: continue
                f = max(self.adj[q].keys(), key=lambda j: self.errors[j] if j < len(self.errors) else 0.0)
                r = 0.5*(np.array(self.nodes[q]) + np.array(self.nodes[f]))
                self._add_node(r[0], r[1]); r_idx = len(self.nodes)-1
                self.adj[q].pop(f, None); self.adj[f].pop(q, None)
                self._add_edge(q, r_idx); self._add_edge(r_idx, f)
                self.errors[q] *= 0.5; self.errors[f] *= 0.5; self.errors[r_idx] = 0.25*(self.errors[q]+self.errors[f])

def run_gng_build(dense_npz: str, out_npz: str):
    data = np.load(dense_npz); pts = data["pts"]
    gng = GNG(); import numpy as np
    idx = np.random.permutation(len(pts)); gng.fit(pts[idx])
    nodes = np.array(gng.nodes, dtype=np.float32)
    edges = np.array([(i,j) for i in gng.adj for j in gng.adj[i] if i<j], dtype=np.int32)
    np.savez_compressed(out_npz, nodes=nodes, edges=edges)
    print(f"[ok] GNG graph -> {out_npz} (nodes={len(nodes)}, edges={len(edges)})")
