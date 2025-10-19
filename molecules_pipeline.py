import sys
import subprocess
import importlib
import traceback

# --------------------
# Helper: install & import safely
# --------------------

def install_package(pip_name):
    """Install a pip package using the current Python executable and return True on success."""
    try:
        print(f"Installing {pip_name} ...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pip_name])
        return True
    except Exception as e:
        print(f"Failed to install {pip_name}: {e}")
        traceback.print_exc()
        return False


def import_or_install(pip_name, module_name=None):
    """Try to import module_name (or pip_name if module_name None). If import fails, pip install pip_name then import."""
    module_name = module_name or pip_name
    try:
        return importlib.import_module(module_name)
    except Exception:
        print(f"Module '{module_name}' not found. Attempting to install '{pip_name}'...")
        ok = install_package(pip_name)
        if not ok:
            raise ImportError(f"Could not install package {pip_name}")
        try:
            return importlib.import_module(module_name)
        except Exception as e:
            print(f"Import failed after installing {pip_name}: {e}")
            raise

try:
    import os, math, time, random
    import numpy as np
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from tdc.generation import MolGen
    from tdc.single_pred import ADME
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    import deepchem as dc
except Exception as e:
    print("Critical import failed. Please check install logs above. Error:")
    traceback.print_exc()
    raise

# Quick check for matplotlib availability (fixes the error reported by user)
try:
    plt.figure()
    plt.plot([0,1],[0,1])
    plt.title('Sanity-check plot (Matplotlib available)')
    # In headless environments, plt.show() may not display; we save the figure to disk instead.
    plt.savefig('sanity_plot.png')
    plt.close()
    print('Matplotlib is available — saved sanity_plot.png')
except Exception as e:
    print('Matplotlib seems not usable in this environment:', e)
    traceback.print_exc()

# --------------------
# Configuration (same as before)
# --------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MODE = 'egnn'   # options: 'egnn' (continuous EGNN denoiser), 'discrete' (birth-death diffusion)
MAX_ATOMS = 38
ATOM_TYPES = ['C','N','O','S','F','Cl','Br','I','P','B']
ATOM_TYPE_TO_IDX = {a:i for i,a in enumerate(ATOM_TYPES)}
NODE_FDIM = len(ATOM_TYPES) + 1
BOND_TYPES = ['SINGLE','DOUBLE','TRIPLE','AROMATIC']
MAX_BOND_TYPES = len(BOND_TYPES)

BATCH_SIZE = 64
LR = 1e-4
EPOCHS = 50          # keep small for quick runs; increase for proper experiments
TIMESTEPS = 200     # reduce for demo; use 1000+ for research
NUM_SAMPLES = 100
DATA_LIMIT = 10000   # limit dataset size for demo

# --------------------
# Utilities: SMILES <-> graph
# --------------------
from rdkit.Chem import rdmolops

def mol_to_graph(smiles, max_atoms=MAX_ATOMS):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    n = mol.GetNumAtoms()
    if n > max_atoms:
        return None
    node = np.zeros((max_atoms, NODE_FDIM), dtype=np.float32)
    for i, a in enumerate(mol.GetAtoms()):
        sym = a.GetSymbol()
        idx = ATOM_TYPE_TO_IDX.get(sym, None)
        if idx is not None:
            node[i, idx] = 1.0
        else:
            node[i, -1] = 1.0
    adj = np.zeros((max_atoms, max_atoms, MAX_BOND_TYPES), dtype=np.float32)
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        bt = b.GetBondType()
        if bt == Chem.rdchem.BondType.SINGLE:
            t = 0
        elif bt == Chem.rdchem.BondType.DOUBLE:
            t = 1
        elif bt == Chem.rdchem.BondType.TRIPLE:
            t = 2
        elif bt == Chem.rdchem.BondType.AROMATIC:
            t = 3
        else:
            continue
        adj[i,j,t] = 1.0
        adj[j,i,t] = 1.0
    return node, adj


def graph_to_smiles(node, adj, atom_thresh=0.3, bond_thresh=0.5):
    mask = node.max(axis=1) > atom_thresh
    n = int(mask.sum())
    if n == 0:
        return None
    rw = Chem.RWMol()
    for i in range(n):
        vec = node[i]
        if vec[:-1].sum() > 0:
            idx = int(np.argmax(vec[:-1]))
            sym = ATOM_TYPES[idx]
        else:
            sym = 'C'
        rw.AddAtom(Chem.Atom(sym))
    for i in range(n):
        for j in range(i+1, n):
            probs = adj[i,j]
            bidx = int(np.argmax(probs))
            val = float(probs[bidx])
            if val > bond_thresh:
                if bidx == 0:
                    bt = Chem.rdchem.BondType.SINGLE
                elif bidx == 1:
                    bt = Chem.rdchem.BondType.DOUBLE
                elif bidx == 2:
                    bt = Chem.rdchem.BondType.TRIPLE
                else:
                    bt = Chem.rdchem.BondType.AROMATIC
                try:
                    rw.AddBond(i, j, bt)
                except:
                    pass
    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
        s = Chem.MolToSmiles(mol)
        return s
    except:
        return None

# --------------------
# Load MOSES (TDC) and prepare tensors (limited for demo)
# --------------------
print('Loading MOSES dataset (this may take a few seconds)...')
mo = MolGen(name='MOSES').get_data()
smiles_all = mo['smiles'].tolist()
print('Total MOSES molecules available:', len(smiles_all))

# prepare small dataset for quick demonstration
graphs = []
valid_smiles = []
for s in smiles_all:
    try:
        g = mol_to_graph(s)
        if g is not None:
            graphs.append(g)
            valid_smiles.append(s)
    except Exception:
        continue
    if len(graphs) >= DATA_LIMIT:
        break
print('Prepared graphs:', len(graphs))

# flatten to vector X = [node_flat, adj_flat]
MAX_N = MAX_ATOMS
NODE_SZ = MAX_N * NODE_FDIM
ADJ_SZ = MAX_N * MAX_N * MAX_BOND_TYPES
TOTAL_DIM = NODE_SZ + ADJ_SZ


def stack_graphs(graphs):
    arr = np.zeros((len(graphs), TOTAL_DIM), dtype=np.float32)
    for i, (node, adj) in enumerate(graphs):
        arr[i,:NODE_SZ] = node.reshape(-1)
        arr[i,NODE_SZ:] = adj.reshape(-1)
    return arr

X = stack_graphs(graphs)
print('X shape:', X.shape)

# --------------------
# Diffusion scheduler (DDPM) - continuous
# --------------------

def linear_beta(timesteps, beta_start=1e-4, beta_end=2e-2):
    return torch.linspace(beta_start, beta_end, timesteps)

betas = linear_beta(TIMESTEPS).to(DEVICE)
alphas = 1.0 - betas
alpha_prod = torch.cumprod(alphas, dim=0)

# helper

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(0, t).reshape(batch_size, *([1]*(len(x_shape)-1))).to(DEVICE)
    return out

# --------------------
# Simple EGNN-style denoiser (adapted) and DiscreteDenoiser
# --------------------
class SimpleEGNN(nn.Module):
    def __init__(self, node_dim, bond_channels, max_nodes=MAX_N, hidden=256):
        super().__init__()
        self.max_nodes = max_nodes
        self.node_dim = node_dim
        self.bond_channels = bond_channels
        # per-node encoder
        self.node_enc = nn.Linear(node_dim, hidden)
        # bond encoder
        self.bond_enc = nn.Linear(bond_channels, hidden)
        # message MLP
        self.msg = nn.Sequential(nn.Linear(2*hidden+1, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        # update
        self.update = nn.GRUCell(hidden, hidden)
        # readout to reconstruct node features and adjacency
        self.node_dec = nn.Sequential(nn.Linear(hidden, node_dim), nn.Softmax(dim=-1))
        self.adj_dec = nn.Sequential(nn.Linear(2*hidden, bond_channels), nn.Softmax(dim=-1))

    def forward(self, x_flat, t):
        B = x_flat.shape[0]
        node_flat = x_flat[:,:NODE_SZ]
        adj_flat = x_flat[:,NODE_SZ:]
        node = node_flat.view(B, self.max_nodes, self.node_dim)
        adj = adj_flat.view(B, self.max_nodes, self.max_nodes, self.bond_channels)
        h = torch.relu(self.node_enc(node))
        b = torch.relu(self.bond_enc(adj))
        H = h.shape[-1]
        h_new = h.clone()
        for i in range(self.max_nodes):
            hj = h
            bij = b[:, :, i, :].view(B, self.max_nodes, H)
            # t as scalar per batch
            t_emb = t.float().unsqueeze(-1).unsqueeze(-1).repeat(1, self.max_nodes, 1) / TIMESTEPS
            msg_in = torch.cat([hj, bij, t_emb], dim=-1)
            m = self.msg(msg_in)
            agg = m.sum(dim=1)
            h_i = h[:, i, :]
            h_new_i = self.update(agg, h_i)
            h_new[:, i, :] = h_new_i
        node_out = self.node_dec(h_new)
        i_rep = h_new.unsqueeze(2).repeat(1,1,self.max_nodes,1)
        j_rep = h_new.unsqueeze(1).repeat(1,self.max_nodes,1,1)
        pair = torch.cat([i_rep, j_rep], dim=-1)
        adj_out = self.adj_dec(pair)
        out_flat = torch.cat([node_out.reshape(B,-1), adj_out.reshape(B,-1)], dim=1)
        return out_flat

class DiscreteDenoiser(nn.Module):
    def __init__(self, input_dim, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_dim)
        )
    def forward(self, x, t):
        t = t.unsqueeze(-1).float()/TIMESTEPS
        t_b = t.repeat(1, x.shape[1])
        inp = torch.cat([x, t_b], dim=1)
        return self.net(inp)

# --------------------
# Choose model
# --------------------
if MODE == 'egnn':
    print('Using EGNN denoiser (continuous DDPM)')
    model = SimpleEGNN(node_dim=NODE_FDIM, bond_channels=MAX_BOND_TYPES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
else:
    print('Using Discrete denoiser (birth-death prototype)')
    model = DiscreteDenoiser(TOTAL_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

# --------------------
# DataLoader
# --------------------
X_tensor = torch.from_numpy(X).to(torch.float32)
# Ensure dataset length is multiple of batch size for drop_last in DataLoader
if len(X_tensor) < BATCH_SIZE:
    raise ValueError(f"Prepared dataset too small ({len(X_tensor)}) for batch size {BATCH_SIZE}. Lower BATCH_SIZE or increase DATA_LIMIT.")

dataset_t = TensorDataset(X_tensor)
loader = DataLoader(dataset_t, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# --------------------
# Training loop
# --------------------
print('Training...')
model.train()
for epoch in range(EPOCHS):
    tot_loss = 0.0
    t0 = time.time()
    for (batch,) in loader:
        batch = batch.to(DEVICE)
        bs = batch.shape[0]
        t_rand = torch.randint(0, TIMESTEPS, (bs,), device=DEVICE).long()
        if MODE == 'egnn':
            noise = torch.randn_like(batch).to(DEVICE)
            alpha_t = extract(alpha_prod, t_rand, batch.shape)
            sqrt_alpha = torch.sqrt(alpha_t)
            sqrt_om = torch.sqrt(1 - alpha_t)
            x_t = sqrt_alpha * batch + sqrt_om * noise
            pred = model(x_t, t_rand)
            loss = loss_fn(pred, noise)
        else:
            prob = (t_rand.float().unsqueeze(1) / TIMESTEPS).to(DEVICE)
            prob = prob.repeat(1, TOTAL_DIM)
            mask = (torch.rand_like(batch) > prob).float()
            corrupted = batch * mask
            logits = model(corrupted, t_rand)
            loss = loss_fn(logits, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    t1 = time.time()
    print(f'Epoch {epoch+1}/{EPOCHS} loss={tot_loss/len(loader):.6f} time={t1-t0:.1f}s')

# --------------------
# Sampling
# --------------------
print('Sampling...')
model.eval()
samples = []
with torch.no_grad():
    while len(samples) < NUM_SAMPLES:
        batch_z = torch.randn(BATCH_SIZE, TOTAL_DIM).to(DEVICE)
        if MODE == 'egnn':
            x = batch_z
            for step in reversed(range(TIMESTEPS)):
                t_step = torch.full((BATCH_SIZE,), step, device=DEVICE, dtype=torch.long)
                pred_noise = model(x, t_step)
                beta = betas[step]
                alpha = alphas[step]
                alpha_p = alpha_prod[step]
                x = (x - beta / torch.sqrt(1 - alpha_p) * pred_noise) / torch.sqrt(alpha)
                if step > 0:
                    x = x + torch.sqrt(beta) * torch.randn_like(x).to(DEVICE)
        else:
            x = torch.rand(BATCH_SIZE, TOTAL_DIM).to(DEVICE)
            for step in reversed(range(TIMESTEPS)):
                t_step = torch.full((BATCH_SIZE,), step, device=DEVICE, dtype=torch.long)
                logits = model(x, t_step)
                probs = torch.sigmoid(logits)
                thresh = 0.5
                x = (probs > thresh).float()
        arr = x.cpu().numpy()
        for v in arr:
            samples.append(v)
        if len(samples) >= NUM_SAMPLES:
            break;
samples = samples[:NUM_SAMPLES]
print('Raw samples generated:', len(samples))

# --------------------
# Convert samples to SMILES
# --------------------
print('Converting to SMILES...')
gen_smiles = []
for vec in samples:
    node_flat = vec[:NODE_SZ]
    adj_flat = vec[NODE_SZ:]
    node = node_flat.reshape((MAX_N, NODE_FDIM))
    adj = adj_flat.reshape((MAX_N, MAX_N, MAX_BOND_TYPES))
    # normalize node rows
    node = np.maximum(node, 0)
    row_sums = node.sum(axis=1, keepdims=True) + 1e-8
    node = node / row_sums
    # softmax adj per pair
    adj = np.maximum(adj, 0)
    pair_sums = adj.sum(axis=2, keepdims=True) + 1e-8
    adj = adj / pair_sums
    s = graph_to_smiles(node, adj, atom_thresh=0.25, bond_thresh=0.5)
    if s is not None:
        gen_smiles.append(s)

# canonicalize & dedup
try:
    gen_smiles = list({Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in gen_smiles if Chem.MolFromSmiles(s) is not None})
except Exception:
    gen_smiles = list(set(gen_smiles))
print('Valid unique generated molecules:', len(gen_smiles))

# --------------------
# Compute QED/logP for generated molecules
# --------------------

def qed_logp(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None, None
    try:
        return Descriptors.qed(m), Descriptors.MolLogP(m)
    except Exception:
        return None, None

rows = []
for s in gen_smiles:
    q,l = qed_logp(s)
    if q is not None:
        rows.append((s,q,l))
gen_df = pd.DataFrame(rows, columns=['smiles','QED','LogP'])
print(gen_df.head())

# --------------------
# ADMET surrogate (Lipophilicity) training & prediction
# --------------------
print('Training lipophilicity surrogate...')
adme = ADME(name='lipophilicity_astrazeneca')
splits = adme.get_split()
train = splits['train']
test = splits['test']

fp = dc.feat.CircularFingerprint(size=1024)
train_X = fp.featurize(train['Drug'])
train_y = train['Y']
test_X = fp.featurize(test['Drug'])
test_y = test['Y']

rf = RandomForestRegressor(n_estimators=200, random_state=SEED)
rf.fit(train_X, train_y)
pred_test = rf.predict(test_X)
print('MAE:', mean_absolute_error(test_y, pred_test), 'R2:', r2_score(test_y, pred_test))

# predict for generated
if len(gen_df) > 0:
    gen_X = fp.featurize(gen_df['smiles'])
    preds = rf.predict(gen_X)
    gen_df['Pred_Lipophilicity'] = preds
    print(gen_df.head())
    gen_df.to_csv('generated_molecules_with_admet.csv', index=False)
    print('Saved generated_molecules_with_admet.csv')
else:
    print('No generated molecules to predict')

# --------------------
# Visualize original vs generated
# --------------------
orig_sample = valid_smiles[:len(gen_df)]
orig_rows = []
for s in orig_sample:
    q,l = qed_logp(s)
    if q is not None:
        orig_rows.append((s,q,l))
orig_df = pd.DataFrame(orig_rows, columns=['smiles','QED','LogP'])

plt.figure(figsize=(8,6))
plt.scatter(orig_df['QED'], orig_df['LogP'], alpha=0.3, label='Original')
if len(gen_df) > 0:
    plt.scatter(gen_df['QED'], gen_df['LogP'], alpha=0.7, label='Generated')
plt.xlabel('QED'); plt.ylabel('LogP'); plt.legend(); plt.title('Original vs Generated')
# save instead of show to avoid environment display issues
plt.savefig('original_vs_generated.png')
print('Saved original_vs_generated.png')
plt.close()

print('Pipeline complete.')

# --------------------
# Quick unit-tests / sanity checks
# --------------------
print('\nRunning quick unit-tests...')

# Test 1: round-trip SMILES -> graph -> SMILES for some simple molecules
_smiles_tests = ['CCO', 'c1ccccc1', 'CC(=O)O']
for s in _smiles_tests:
    g = mol_to_graph(s)
    assert g is not None, f'Failed to convert {s} to graph'
    node, adj = g
    s2 = graph_to_smiles(node, adj)
    assert s2 is not None, f'Round-trip failed for {s}, got None'
    print(f'Round-trip OK: {s} -> {s2}')

# Test 2: matplotlib file exists
import os
assert os.path.exists('sanity_plot.png'), 'sanity_plot.png not found. Matplotlib may not be available.'
print('Matplotlib sanity plot present. Unit-tests passed.')

print('\nIf anything failed, read the installation logs above. If you want me to tailor the environment (e.g., use a CUDA PyTorch wheel), tell me which environment you are using (Colab / local / HPC) and I will adapt the install commands.')
