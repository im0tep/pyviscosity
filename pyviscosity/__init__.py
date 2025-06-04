import numpy as np 
import pandas as pd 
import rdkit
import cirpy
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw, MolFromSmiles
from rdkit.Chem import PeriodicTable
from rdkit.Chem import GetPeriodicTable
from pkg_resources import resource_filename
import re
from functools import reduce
import inspect
#import Hsu method data 
names= ['Q','Group','ai','bi','ci','di']
resdata = resource_filename('pyviscosity', 'group_data.csv')
group_data = pd.read_csv(resdata,sep=',', header=0, names=names)

#import direct mapping from Hsu functional groups to Joback 
names=['Q', 'jb']
mapdata = resource_filename('pyviscosity', 'joback.map.csv')
map_data= pd.read_csv(mapdata, index_col=False, names=names)

#import joback datasets 
# nr= group from Hsu method
# Tc = Critical Temperature, Pc= Critical Pressure, Vc= Critical Volume, Tb= Normal boiling point, Tf= Normal freezing point,
# H= Enthalpy of formation ideal gas at 298K, G= Gibbs ernergy of formation ideal gas unit fugacity at 298K
# a/b/c/d= ideal gas heat capacity, vap= enthalpy of vaporisation at Tb, fus= enthalpy of fustion, etaA/etaB= liquid viscosity
joback = [ resource_filename('pyviscosity', 'joback.'+i+'.csv') for i in ['1','2','3','4']] 
joback_cols = [['nr','grp','Tc','Pc','Vc'],['nr','Tb','Tf','H','G'],['nr','a','b','c','d'],['nr','vap','fus','etaA','etaB'] ]
joback_data = [pd.read_csv(joback[j], sep='\s+', index_col=False, names=joback_cols[j]) for j in range(len(joback))]
for j in joback_data: j['id']=range(len(j))
joback_dfs= [joback_data[i] for i in range(4)]
joback_merged = reduce(lambda  left,right: pd.merge(left,right,on=['id', 'nr'],
                                            how='inner'), joback_dfs)

PT = GetPeriodicTable()


class Molecule(object):

    def __init__(self, inp,  debug=False):
        if len(re.findall(r'\b[1-9]{1}[0-9]{1,5}-\d{2}-\d\b',inp)) > 0:
            if debug: print('it\'s a CAS:',inp)
            self.smile = cirpy.resolve(inp,'smiles')
            if debug: print('and it translates to',self.smile)
        else:
            if debug: print('it\'s a smiles:',inp)
            self.smile = inp    
        self.mol= Chem.MolFromSmiles(self.smile)
        self._Qs = []
        self.processed = set()
        # here starts the function
        self.atoms = self.mol.GetAtoms()
        self.matrix = rdkit.Chem.rdmolops.GetDistanceMatrix(self.mol)
        self.elements = np.array([PeriodicTable.GetElementSymbol(PT, at.GetAtomicNum()) for at in self.atoms]) 
        functions = [name for name, value in Molecule.__dict__.items() if callable(value) and not name.startswith("__")]
        for attribute in functions:#dir(self):
                if 'check_' in attribute: getattr(self, attribute)()
                if '_check' in attribute: getattr(self, attribute)()

    def count_atoms(self):
        # Create a molecule object from the SMILES string
        molecule = Chem.MolFromSmiles(self.smile)
        
        # Add explicit hydrogens to the molecule
        molecule_with_hydrogens = Chem.AddHs(self.mol)
        
        # Count the total number of atoms
        total_atoms = molecule_with_hydrogens.GetNumAtoms()
        
        return total_atoms

    @property
    def critical_pressure(self):
        _Pc= .0 
        for i in self._Qs: 
            jb= map_data.loc[map_data['Q'] == i, 'jb'].item().split()
            for j in jb: 
                _Pc += float(joback_merged.loc[joback_merged['id'] == int(j), 'Pc'].iloc[0]) 
        n_A= self.count_atoms() 
        return (0.113 + 0.0032*n_A - _Pc)**(-2) 

    def viscosity(self,T): 
        a= [i[0] for i in self.parameters]
        b= [i[1] for i in self.parameters]
        c= [i[2] for i in self.parameters]
        d= [i[3] for i in self.parameters]
        Pc= self.critical_pressure
        return np.exp(sum(a) + 0.01*T*sum(b) + 10000*sum(c)/(T**2) + sum(d)*np.log(Pc))

    @property
    def functional_groups_hsu(self):
        return sorted(self._Qs)

    @property
    def functional_groups_joback(self):
        joback_fgs= []
        for i in self._Qs:
            jb= map_data.loc[map_data['Q'] == i, 'jb'].item().split()
            for j in jb:
                joback_fgs.append(j)
        return joback_fgs

    @property
    def parameters(self):
        parms=[]
        for Q in sorted(self._Qs):
            try: p = group_data[group_data.Q.values==Q][['ai','bi','ci','di']].values[0].tolist()
            except: p = None
            parms.append(p)
        return parms

    def GetRingSystems(self,includeSpiro=False):
        ri = self.mol.GetRingInfo()
        systems = []
        for ring in ri.AtomRings():
            ringAts = set(ring)
            found = False
            nSystems = []
            for system in systems:
                nInCommon = len(ringAts.intersection(system)) 
                if nInCommon and (includeSpiro or nInCommon>1):
                    ringAts = ringAts.union(system)
                else:
                    nSystems.append(system)
            nSystems.append(ringAts)
            systems = nSystems
        return systems

    def isRingAromatic(self, bondRing):
        for id in bondRing:
            if not self.mol.GetBondWithIdx(id).GetIsAromatic():
                return False
        return True

    def isAtomOnRing(self, atom):
        rings= self.GetRingSystems()
        ring_info=self.mol.GetRingInfo()
        ring= False
        arom=False
        direct=False
        indices = np.nonzero(self.matrix[atom]==1)[0] #get positions of bonded atoms 
        for i,ri in enumerate(rings): #loop through rings 
            if any(x in ri for x in indices) is True: #if atom m is bonded directly to an atom in a ring return true 
                ring= True 
                arom= self.isRingAromatic(ring_info.BondRings()[i]) #if this ring is aromatic return true 
            if sum(x in ri for x in indices) >= 2:
                direct= True
        # want to return true if bonded to any atom in any ring, the aromaticity of that ring and whether the atom is directly on the ring or bonded to it        
        return ring, arom, direct 

    def contains_alkoxylalcohol(self):
        # Create the pattern molecules
        alkoxy_pattern = Chem.MolFromSmarts("[OD2]([#6])[#6]")
        alcohol_pattern = Chem.MolFromSmarts("[CX4][OX2H]" )
        # Check if the molecule contains both an alkoxy group and an alcohol group
        contains_alkoxy = self.mol.HasSubstructMatch(alkoxy_pattern)
        contains_alcohol = self.mol.HasSubstructMatch(alcohol_pattern)
        return contains_alkoxy and contains_alcohol
   
    def check_anyhydride(self):
        sub = Chem.MolFromSmarts('C(=O)OC=O')
        match = list(self.mol.GetSubstructMatches(sub))
        for m in match:
            if any(e in self.processed for e in m): continue
            self._Qs.append(42)
            self.processed = self.processed | set(m)

    def check_acyl_chloride(self):
        sub = Chem.MolFromSmarts('[$(C(=O)Cl)]')
        match = list(self.mol.GetSubstructMatches(sub))
        for m in match:
            if any(e in self.processed for e in m): continue
            self._Qs.append(91)
            indices = np.nonzero(self.matrix[m[0]]==1)[0]
            ind_Cl_O= [i for i in indices if self.elements[i] == 'Cl' or self.elements[i] == 'O']
            self.processed = self.processed | set(m)
            self.processed = self.processed | set(ind_Cl_O)

    def check_carbonate(self):
        sub = Chem.MolFromSmarts('[O-]C([O-])=O')
        match = list(self.mol.GetSubstructMatches(sub)) 
        for m in match:
            if any(e in self.processed for e in m): continue
            self._Qs.append(43)
            self.processed = self.processed | set(m)

    def check_carboxylic_acid(self): 
        sub = Chem.MolFromSmarts('[CX3](=O)[OX2H1]')
        match = list(self.mol.GetSubstructMatches(sub)) 
        for m in match:  
            if any(e in self.processed for e in m): continue
            indices = np.nonzero(self.matrix[m[0]]==1)[0]
            if self.elements[indices].size == 2: self._Qs.append(35)
            if self.elements[indices].size == 3 and np.sum(self.elements=='C') <=6 : self._Qs.append(36)
            if self.elements[indices].size == 3 and np.sum(self.elements=='C') >6 : self._Qs.append(37)
            self.processed = self.processed | set(m) 
         
    def check_formate_ion(self):
        sub = Chem.MolFromSmarts('[CX3H1](=O)[OX2]')
        match = list(self.mol.GetSubstructMatches(sub))
        for m in match:
            if any(e in self.processed for e in m): continue
            indices = np.nonzero(self.matrix[m[0]]==1)[0]
            if self.elements[indices].size == 2: self._Qs.append(38)
            self.processed = self.processed | set(m)


    def check_carboxylate_ion(self):  
        sub = Chem.MolFromSmarts('[CX3](=O)[OX2]')
        match = list(self.mol.GetSubstructMatches(sub)) 
        for m in match:  
            if any(e in self.processed for e in m): continue
            indices = np.nonzero(self.matrix[m[0]]==1)[0]
            if self.elements[indices].size == 3 and np.sum(self.elements=='C') <=7 : self._Qs.append(39)
            if self.elements[indices].size == 3 and np.sum(self.elements=='C') >7 : self._Qs.append(40)
            self.processed = self.processed | set(m)

    def check_carbamate(self):
        sub= Chem.MolFromSmarts('[OX2][CX3H0](=[OX1H0])[NX3]')
        match = list(self.mol.GetSubstructMatches(sub)) 
        for m in match: 
            if any(e in self.processed for e in m): continue
            for n in m: 
                for el in (self.elements[n]):
                    if el == 'N':
                        indices = np.nonzero(self.matrix[n]==1)[0]
                        n_ind = self.elements[indices].size 
            if n_ind == 1: self._Qs.append(66) 
            elif n_ind == 2 : self._Qs.append(67) 
            self.processed = self.processed | set(m)

    def check_nitrile(self): 
        sub= Chem.MolFromSmarts('[NX1]#[CX2]')
        match = list(self.mol.GetSubstructMatches(sub)) 
        for m in match: 
            if any(e in self.processed for e in m): continue
            for n in m:
                if self.elements[n] == 'C':
                    ring, arom= self.isAtomOnRing(n)[0:2]
                    if ring is False:
                        self._Qs.append(70)
                     #checkkkk logic 
                    else: 
                        if arom is True: self._Qs.append(71)
            self.processed= self.processed | set(m)     
            
    def check_amide(self): 
        sub= Chem.MolFromSmarts('[NX3][CX3](=[OX1])')
        match = list(self.mol.GetSubstructMatches(sub)) 
        c_ind, n_ind= None, None  
        for m in match: 
            if any(e in self.processed for e in m): continue
            for n in m: 
                for el in (self.elements[n]):
                    if el == 'N':
                        indices = np.nonzero(self.matrix[n]==1)[0]
                        n_ind = self.elements[indices].size 
                    elif el == 'C':
                        indices = np.nonzero(self.matrix[n]==1)[0]
                        c_ind = self.elements[indices].size 
        if c_ind == 2 and n_ind ==1: 
            self._Qs.append(61) 
            self.processed = self.processed | set(m)
        elif c_ind == 2 and n_ind ==2: 
            self._Qs.append(62) 
            self.processed = self.processed | set(m)
        elif c_ind == 2 and n_ind ==3: 
            self._Qs.append(63) 
            self.processed = self.processed | set(m)
        elif c_ind == 3 and n_ind ==1: 
            self._Qs.append(64) 
            self.processed = self.processed | set(m)
        elif c_ind == 3 and n_ind ==2: 
            self._Qs.append(65) 
            self.processed = self.processed | set(m)

    def check_thioester(self): #add thionoester?  
        sub = Chem.MolFromSmarts('S([#6])[CX3](=O)[#6]')
        match = list(self.mol.GetSubstructMatches(sub))
        for m in match:
            ind_S= [n for n in m if self.elements[n]=='S']
            ind_O= [n for n in m if self.elements[n]=='O']
            indices = np.nonzero(self.matrix[ind_O[0]]==1)[0]
            ind_C= [n for n in m if self.elements[n]=='C' and n in indices]
            if ind_S[0] in self.processed: continue
            if np.sum(self.elements=='C') <=12 : self._Qs.append(52)
            if np.sum(self.elements=='C') >12 : self._Qs.append(53)
            self.processed = self.processed | set({ind_S[0]})
            self.processed = self.processed | set({ind_C[0]})
            self.processed = self.processed | set({ind_O[0]})

    def check_thioether(self): 
        sub = Chem.MolFromSmarts('[#16;X2]') #[CX3H1](=O)[#6]
        match = list(self.mol.GetSubstructMatches(sub))
        for m in match:
            if any(e in self.processed for e in m): continue
            for n in m:
                if self.elements[n] == 'S':
                    indices = np.nonzero(self.matrix[n]==1)[0]
                    if len(indices) == 2:
                        self._Qs.append(48)
                        self.processed = self.processed | set({n})

    def check_sulfinyl(self): 
        sub = Chem.MolFromSmarts('[$([#16X3]=[OX1]),$([#16X3+][OX1-])]')
        match = list(self.mol.GetSubstructMatches(sub))
        for m in match:
            if any(e in self.processed for e in m): continue
            indices= np.nonzero(self.matrix[m]==1)[0]
            ind_O= [n for n in indices if self.elements[n]=='O'] 
            processed = self.processed | set(m)
            processed = self.processed | set(ind_O)
            self._Qs.append(54)

    def check_ether_secondary(self): 
        #alkoxy group where the carbon attached to the oxygen is also attached to 2 carbon chains... just a special case ether? Check 
        sub = Chem.MolFromSmarts('[OD2]([#6])[#6]')
        match = list(self.mol.GetSubstructMatches(sub))
        for m in match:
            if any(e in self.processed for e in m): continue
            C_chain= []
            C_ind= []
            for n in m: 
                if self.elements[n] == 'C': 
                    indices = np.nonzero(self.matrix[n]==1)[0]
                    hybridization = str(self.atoms[n].GetHybridization()) 
                    noCs= [x for x in indices if self.elements[x]=='C']
                    ind_O= [x for x in indices if self.elements[x]=='O']
                    if len (indices) == 3 and len(noCs) == 2 and hybridization == 'SP3': 
                        C_chain.append(True)
                        C_ind.append(n) 
            if any(C_chain): 
                self._Qs.append(41)
                self.processed= self.processed | set({ind_O[0]}) 
                self.processed= self.processed | set({C_ind[0]}) #only want to process one C if there's 2 - the other will be processed seperately 

    def check_ether(self):  
        sub = Chem.MolFromSmarts('[OD2]([#6])[#6]')
        match = list(self.mol.GetSubstructMatches(sub)) 
        for m in match:
            ind_O= [n for n in m if self.elements [n]=='O'] #classify as ether even if carbon has already been self.processed e.g. as above 35655-96-0
            if ind_O[0] in self.processed: continue
            for n in m:
                if self.elements [n] == 'O':
                    ring, arom= self.isAtomOnRing(n)[0:2]
                    if ring is False:
                        self._Qs.append(29)
                     #checkkkk logic
                    else:
                        if arom is False: self._Qs.append(30)
                        if arom is True: self._Qs.append(31)
                    self.processed= self.processed | set({n})
    
    def check_direct_aromatic_ether(self):
        sub = Chem.MolFromSmarts('c:o:c')
        match = list(self.mol.GetSubstructMatches(sub)) 
        for m in match:
            ind_O= [n for n in m if self.elements[n]=='O'] #classify as ether even if carbon has already been processed e.g. as above 35655-96-0
            if ind_O[0] in self.processed: continue
            for n in m:
                if self.elements[n] == 'O':
                    ring, arom= self.isAtomOnRing(n)[0:2]
                    if arom is True: self._Qs.append(31)
                    self.processed= self.processed | set({n}) 

    def check_aldehyde(self):  
        sub = Chem.MolFromSmarts('[CX3H1](=O)') #[CX3H1](=O)[#6]
        match = list(self.mol.GetSubstructMatches(sub)) 
        for m in match:
            if any(e in self.processed for e in m): continue
            self._Qs.append(32)
            self.processed = self.processed | set(m)

    def check_ketone(self):
        sub = Chem.MolFromSmarts('[CX3](=O)') #'[#6][CX3](=O)[#6]'
        match = list(self.mol.GetSubstructMatches(sub)) 
        for m in match: 
            if any(e in self.processed for e in m): continue
            for n in m: 
                if self.elements[n] == 'O':
                    ring, arom= self.isAtomOnRing(n)[0:2]
                    if ring is False:
                        self._Qs.append(33)
                        self.processed = self.processed | set(m)
                    if ring is True:
                        self._Qs.append(34)
                        self.processed = self.processed | set(m)
 
    def check_alkoxyalcohol(self): 
        if self.contains_alkoxylalcohol() == True:
            sub = Chem.MolFromSmarts('[OH1]')
            match = list(self.mol.GetSubstructMatches(sub))
            for m in match:
                if m[0] in self.processed : continue
                self._Qs.append(28)
                self.processed = self.processed | set(m)

    def check_alcohol(self):
        sub = Chem.MolFromSmarts('[OH1]')
        match = list(self.mol.GetSubstructMatches(sub))
        for m in match:
            if m[0] in self.processed : continue
            ring, arom= self.isAtomOnRing(m)[0:2]
            if ring is False:
                if len (match) == np.sum(self.elements =='C') and len(match) > 1:
                    self._Qs.extend([26])
                    self.processed = self.processed | set(m)
                else:
                    indices = np.nonzero(self.matrix [m[0]]==2)[0]
                    if self.elements[indices].size == 0 and np.sum(self.elements=='C') == 1 : self._Qs.append(21)
                    if self.elements [indices].size == 1 and np.sum(self.elements=='C') <=2 : self._Qs.append(21)
                    if self.elements [indices].size == 1 and np.sum(self.elements=='C') >2 : self._Qs.append(22)
                    elif self.elements [indices].size == 2: self._Qs.append(23)
                    elif self.elements [indices].size == 3: self._Qs.append(24)
                    self.processed = self.processed | set(m)
            else:
                if arom is False: self._Qs.append(25)
                if arom is True: self._Qs.append(27)
                self.processed = self.processed | set(m)

    def check_nitroalkene(self): 
        sub = Chem.MolFromSmarts('[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]')
        match = list(self.mol.GetSubstructMatches(sub))
        for m in match:
            if any(e in self.processed for e in m): continue
            for n in m:
                if self.elements[n] == 'N':
                    indices = np.nonzero(self.matrix[n]==1)[0]
                    ind_C= [i for i in indices if self.elements[i] == 'C'][0]
                    ind_O= [i for i in indices if self.elements[i] == 'O']
                    hybridization = str(self.atoms[int(ind_C)].GetHybridization())
                    if hybridization == 'SP2':
                        self._Qs.append(46)
                        self.processed = self.processed | set({ind_C})
                        self.processed = self.processed | set(ind_O)
                        self.processed = self.processed | set({n})

    def check_nitro(self):  
        sub = Chem.MolFromSmarts('[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]')
        match = list(self.mol.GetSubstructMatches(sub))
        for m in match:
            if any(e in self.processed for e in m): continue
            for n in m: 
                if self.elements[n] == 'N':  
                    ring, arom= self.isAtomOnRing(n)[0:2]
                    indices = np.nonzero(self.matrix[n]==1)[0]
                    ind_O= [i for i in indices if self.elements[i] == 'O']
                    if ring is False:
                        self._Qs.append(45)
                        self.processed = self.processed | set(ind_O)
                        self.processed = self.processed | set({n})
                    if ring is True and arom is True:
                        self._Qs.append(47)
                        self.processed = self.processed | set(ind_O)
                        self.processed = self.processed | set({n})

    def check_nitroso(self): ####the nitrogen is not bonded to 2 carbons here, check
        sub = Chem.MolFromSmarts('[*][NX2]=[OX1H0]') #[CX3H1](=O)[#6]
        match = list(self.mol.GetSubstructMatches(sub))
        for m in match:
            if any(e in self.processed for e in m): continue
            for n in m:
                if self.elements[n] == 'N':
                    ring, arom= self.isAtomOnRing(n)[0:2]
                    if ring is True and arom is False:
                        self._Qs.append(44)
                        self.processed = self.processed | set(m)

    def check_mercaptan(self): 
        sub = Chem.MolFromSmarts('[#16X2H]') #[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]'
        match = list(self.mol.GetSubstructMatches(sub)) 
        for m in match:
            if m in self.processed : continue
            indices = np.nonzero(self.matrix[m[0]]==2)[0]
            if self.elements[indices].size == 0 or self.elements[indices].size == 1 : self._Qs.append(49)
            elif self.elements[indices].size == 2 : self._Qs.append(50)
            elif self.elements[indices].size == 3: self._Qs.append(51)   
            self.processed = self.processed | set(m)  
  
    def check_primary_amine(self): 
        sub=  Chem.MolFromSmarts('[N&X3;H2]') 
        match = list(self.mol.GetSubstructMatches(sub)) 
        for m in match:
            if any(e in self.processed for e in m): continue
            for n in m: 
                if self.elements[n] == 'N':
                    ring, arom= self.isAtomOnRing(n)[0:2]
                    if ring is False:
                        self._Qs.append(55)  #primary amine -NH2
                        self.processed = self.processed | set({n}) 
                    elif ring is True and arom is False:
                        self._Qs.append(55)  #primary amine -NH2 on non aromatic ring 
                        self.processed = self.processed | set({n})
                    elif ring is True and arom is True: 
                        self._Qs.append(58)  #primary aromatic amine (-NH2)_AC
                        self.processed = self.processed | set({n})       

    def check_secondary_amine(self): 
        sub=  Chem.MolFromSmarts('[H1;!$(NC=*)]') 
        match = list(self.mol.GetSubstructMatches(sub)) 
        for m in match:
            if m in self.processed : continue
            for n in m: 
                if self.elements[n] == 'N':
                    ring, arom= self.isAtomOnRing(n)[0:2]
                    if ring is False:
                        self._Qs.append(56)  #secondary amine -NH-
                        self.processed = self.processed | set({n}) 
                    elif ring is True and arom is True: 
                        self._Qs.append(59)  #secondary aromatic amine (-NH-)_AC
                        self.processed = self.processed | set({n})
                    elif ring is True and arom is False: 
                        self._Qs.append(68)  #secondary amine on ring (>NH)_R
                        self.processed = self.processed | set({n})
                        
    def check_tertiary_amine(self):                
        sub = Chem.MolFromSmarts('[#6]-[#7](-[#6])-[#6]') 
        match = list(self.mol.GetSubstructMatches(sub)) 
        for m in match:
            for n in m: 
                if self.elements[n] == 'N':
                    if n in self.processed: continue 
                    ring, arom= self.isAtomOnRing(n)[0:2]
                    if ring is False:
                        self._Qs.append(57)  #tertiary amine -NH<
                        self.processed = self.processed | set({n}) 
                    elif ring is True and arom is True: 
                        self._Qs.append(60)  #tertiary aromatic amine (-NH<)_AC
                        self.processed = self.processed | set({n})
    #ring systems
    def check_cyclic_imine(self):
        sub = Chem.MolFromSmarts('c:n:c') 
        match = list(self.mol.GetSubstructMatches(sub)) 
        for m in match:
            if any(e in self.processed for e in m): continue
            for n in m: 
                if self.elements[n] == 'N':
                    ring, arom, direct = self.isAtomOnRing(n)
                    if ring is True and arom is True and direct is True: 
                        self._Qs.append(69) # (=N-)_R  
                        self.processed = self.processed | set({n}) 

    def check_biphenyl(self): 
        sub = Chem.MolFromSmarts('c1ccc(cc1)c2ccccc2')
        match = list(self.mol.GetSubstructMatches(sub)) 
        for m in match:
            if m in self.processed : continue
            self.processed = self.processed | set(m)
            self._Qs.extend([17]*2) 
            self._Qs.extend([15]*10)#how many do we add for biphenyl? 

    def check_terphenyl(self):
        subs= ['c1ccc(cc1)c2ccc(cc2)c3ccccc3', 'c1ccc(cc1)c2cccc(c2)c3ccccc3', 'c1ccc(cc1)c2ccccc2c3ccccc3'] #para, meta and orpho
        for s in subs: 
            sub= Chem.MolFromSmarts(s)
            match = list(self.mol.GetSubstructMatches(sub)) 
            for m in match:
                if m in self.processed : continue
                self.processed = self.processed | set(m)
                self._Qs.extend([17]*4)
                self._Qs.extend([15]*14)

    def check_naphalene(self): 
        sub = Chem.MolFromSmarts('c1cccc(c12)cccc2')
        match = list(self.mol.GetSubstructMatches(sub)) 
        for m in match:
            if m in self.processed : continue
            self.processed = self.processed | set(m)
            self._Qs.extend([18]*2) 
            self._Qs.extend([15]*8)


    def check_tetralin(self): 
        sub = Chem.MolFromSmarts('C1CCc2ccccc2C1') #this is smile not smarts...
        match = list(self.mol.GetSubstructMatches(sub)) 
        for m in match:
            if m in self.processed : continue
            self.processed = self.processed | set(m)
            self._Qs.extend([20]*2)
            self._Qs.extend([15]*4)
            self._Qs.extend([11]*4)
     
    def check_carbon_FCl_alkene_Cl(self): #check carbon groups with FCl or =CHCl 
        for i, el in enumerate(self.elements):
            if i in self.processed : continue
            if el == 'C':
                hybridization = str(self.atoms[i].GetHybridization())
                if hybridization == 'SP3':
                    indices = np.nonzero(self.matrix[i]==1)[0]
                    ###### groups with FCl
                    els= self.elements[indices]
                    if all(x in els for x in ['Cl', 'F']):
                        ind_F= [i for i in indices if self.elements[i] == 'F']
                        ind_Cl= [i for i in indices if self.elements[i] == 'Cl']
                        count_F= len (ind_F)
                        count_Cl= len (ind_Cl)
                        if count_F == 1 and count_Cl == 1 : self._Qs.append(82)
                        elif count_F == 1 and count_Cl == 2 : self._Qs.append(83)
                        elif count_F == 2 and count_Cl == 1 : self._Qs.append(84)
                        elif count_F == 2 and count_Cl == 2 : self._Qs.append(85)
                        self.processed = self.processed | set(ind_F)
                        self.processed = self.processed | set(ind_Cl)
                elif hybridization == 'SP2':
                    indices = np.nonzero(self.matrix[i]==1)[0]
                    ###### groups with =CHCl
                    els= self.elements[indices]
                    if 'Cl' in els and len (els) ==2:
                        self._Qs.append(73)
                        ind_Cl=[i for i in indices if self.elements[i] == 'Cl']
                        self.processed = self.processed | set({i})
                        self.processed = self.processed | set(ind_Cl)

    # Here we check single atoms which are not in the listed groups
    def check_unlisted(self): 
        for i, el in enumerate(self.elements):
            if i in self.processed : continue
            indices = np.nonzero(self.matrix[i]==1)[0]
            indices_2 = np.nonzero(self.matrix[i]==2)[0]
            if el == 'C':
                ring, arom, direct= self.isAtomOnRing(i)
                hybridization = str(self.atoms[i].GetHybridization())
                if direct is False:
                    if hybridization == 'SP3':
                        if self.elements[indices].size == 0: self._Qs.append(1)
                        elif self.elements[indices].size == 1: self._Qs.append(2)
                        elif self.elements[indices].size == 2: self._Qs.append(3)
                        elif self.elements[indices].size == 3: self._Qs.append(4)
                        elif self.elements[indices].size == 4: self._Qs.append(5)
                        self.processed = self.processed | set({i})
                    elif hybridization == 'SP2':
                        if self.elements[indices].size == 1: self._Qs.append(6)
                        elif self.elements[indices].size == 2: self._Qs.append(7)
                        elif self.elements[indices].size == 3: self._Qs.append(8)
                        self.processed = self.processed | set({i})
                    elif hybridization == 'SP':
                        if self.elements[indices].size == 1: self._Qs.append(9)
                        elif self.elements[indices].size == 2: self._Qs.append(10)
                        self.processed = self.processed | set({i})
                if ring is True and direct is True and arom is False:
                    if hybridization == 'SP3':
                        if self.elements[indices].size == 2: self._Qs.append(11)
                        elif self.elements[indices].size == 3: self._Qs.append(12)
                        elif self.elements[indices].size == 4: self._Qs.append(14)
                        self.processed = self.processed | set({i})
                    if hybridization == 'SP2':
                        if self.elements[indices].size == 2: self._Qs.append(13)
                        self.processed = self.processed | set({i})
                if ring is True and direct is True and arom is True:
                    if self.elements[indices].size == 2: self._Qs.append(15)
                    elif self.elements[indices].size == 3: self._Qs.append(16)
                    self.processed = self.processed | set({i})
            elif el == 'Cl':
                ring, arom= self.isAtomOnRing(i)[0:2]
                if ring is False:
                    ind_Cl= [i for i in indices_2 if self.elements[i] == 'Cl']
                    count= len(ind_Cl)+1
                    if   count == 1 : self._Qs.append(72)
                    elif   count == 2 : self._Qs.append(74)
                    elif count == 3 : self._Qs.append(75)
                    elif count == 4 : self._Qs.append(76)
                    self.processed = self.processed | set({i})
                    self.processed = self.processed | set(ind_Cl)
                elif ring is True and arom is True:
                    self._Qs.append(77)
                    self.processed = self.processed | set({i})
            elif el == 'F':
                ring, arom= self.isAtomOnRing(i)[0:2]
                if ring is False:
                    ind_F= [i for i in indices_2 if self.elements[i] == 'F']
                    count= len(ind_F)+1
                    if   count == 1 : self._Qs.append(78)
                    elif count == 2 : self._Qs.append(79)
                    elif count == 3 : self._Qs.append(80)
                    self.processed = self.processed | set({i})
                    self.processed = self.processed | set(ind_F)
                elif ring is True and arom is True:
                    self._Qs.append(81)
                    self.processed = self.processed | set({i})
            elif el == 'Br':
                ring, arom= self.isAtomOnRing(i)[0:2]
                if ring is False:
                    count = len(indices_2) # primary vs secondary not 1 vs 2 
                    if   count == 0 or count == 1 : self._Qs.append(86)
                    elif count == 2 : self._Qs.append(87)
                    self.processed = self.processed | set({i})
                elif ring is True and arom is True:
                    self._Qs.append(88)
                    self.processed = self.processed | set({i})
            elif el == 'I':
                ring, arom= self.isAtomOnRing(i)[0:2]
                if ring is False:
                    count = len(indices_2)
                    if   count == 0 or count == 1 : self._Qs.append(89)
                    self.processed = self.processed | set({i})
                elif ring is True and arom is True:
                    self._Qs.append(90)
                    self.processed = self.processed | set({i})


    def proccessing_check(self): 
        if len (self.processed) == self.mol.GetNumAtoms(): 
          print ('all atoms in the molecule have been processed') 
        else: 
          print ('there are some atoms missing from processing')

    def _tests(self):
        """
            >>> from pyviscosity import Molecule
            >>> M = Molecule('98-08-8') #Benzotrifluoride
            >>> M.functional_groups
            [5 15 16 80]
        """
        pass

#for lab in labels:
 #   print(group_data[group_data.Group.str == lab])


if __name__ == '__main__' :
    import argparse
    parser = argparse.ArgumentParser(
                    prog='pyviscosity',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    parser.add_argument('input')
    print(parser.input)
