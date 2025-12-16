# pyviscosity

This repo contains a tool to predict shear viscosities directly from input CAS numbers or SMILES strings. Using RDKit and NumPy, it assigns functional groups via chemical logic (e.g. distinguishing alcohols, ethers, amides) to obtain Hsu/Joback group contributions and estimate viscosities, providing a practical cheminformatics tool for benchmarking molecular dynamics simulations when experimental data are sparse.


# example usage 
```python
from pyviscosity import Molecule
from rdkit import Chem
from rdkit.Chem import Draw

smiles = "CCOC(=O)OCC"
mol = Molecule(smiles)

visc = mol.viscosity(298.15)
fgs_hsu = mol.functional_groups_hsu
fgs_joback = mol.functional_groups_joback
cp = mol.critical_pressure

print("Viscosity (298 K):", visc)
print("Hsu groups:", fgs_hsu)
print("Joback groups:", fgs_joback)
print("Critical pressure:", cp)

rdmol = Chem.MolFromSmiles(smiles)
Draw.MolsToGridImage([rdmol], subImgSize=(200, 200))
```

# SSL certificates for MacOS

In some cases, to use the urlib library within cirpy, one first need to update the SSL certificates.
There are scripts within the /Library directory, find them with:
    mdfind -name "Install Certificates"
    /Applications/Python 3.12/Install Certificates.command
    /Applications/Python 3.11/Install Certificates.command
    /Applications/Python 3.7/Install Certificates.command
    /Applications/Python 3.8/Install Certificates.command


Then execute those scripts (some might be failing, if the corresponding python version is not installed, but this is not a problem)

