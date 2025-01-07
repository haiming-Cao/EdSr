import json, os

"""
thermo_style = [
    # energy
    "pe", # total potential energy
    "ke", # kinetic energy
    "etotal", # total energy (pe + ke)
    "evdwl", # van der Waals pairwise energy (includes etail)
    "ecoul", # Coulombic pairwise energy
    "epair", # pairwise energy (evdwl + ecoul + elong)
    "ebond", # bond energy
    "eangle", # angle energy
    "edihed", # dihedral energy
    "eimp", # improper energy
    "emol", # molecular energy (ebond + eangle + edihed + eimp)
    "elong", # long-range kspace energy
    "etail", # van der Waals energy long-range tail correction
    "enthalpy", # enthalpy (etotal + press*vol)
    "ecouple", # cumulative energy change due to thermo/baro statting fixes
    "econserve", # pe + ke + ecouple = etotal + ecouple
    # properties
    "atoms", # number of atoms
    "temp", # temperature
    "press", # pressure
    "vol", # volume
    "density", # mass density of system
    lx,ly,lz = box lengths in x,y,z
]
"""

def data_parser(file_name: str, check_force: bool = False) -> dict:
    try:
        assert file_name is not None
        with open(file_name, 'r', encoding = 'utf-8') as infile:
            default_params = json.load(infile)
        
        print(f"successfully load json file {os.path.abspath(file_name)}")
    except:
        if check_force:
            raise FileNotFoundError(f'{file_name} not found.')
        
        params = {
            'basis'      : 1.0,
            'ntimestep'  : 10,
            'ensemble'   : "nve",
            'thermo'     : 1,

            'maxIter'    : 500,
            'ntrajs'     : 30,
            'mode'       : "EdSr",
            'prerun_step': 0,

            "fdname"     : None,
            "savepath"   : None,
            'prefix'     : None,
            'split'      : 20,
            'drop_last'  : 0,

            'lmpfile'    : None,

            'properties_saving': [
                'temp', # temperature
                'press', # pressure
                "pe", # total potential energy
                "ke", # kinetic energy
                "etotal", # total energy (pe + ke)
                "evdwl", # van der Waals pairwise energy (includes etail)
                "ecoul", # Coulombic pairwise energy
                "epair", # pairwise energy (evdwl + ecoul + elong)
                "ebond", # bond energy
                "eangle", # angle energy
                "edihed", # dihedral energy
                "eimp", # improper energy
                "emol", # molecular energy (ebond + eangle + edihed + eimp)
                "elong", # long-range kspace energy
                "etail", # van der Waals energy long-range tail correction
                "enthalpy", # enthalpy (etotal + press*vol)
                "ecouple", # cumulative energy change due to thermo/baro statting fixes
            ],

            # unlike properties_saving, log file only displays properties in the thermo_output 
            # and don't save to npz file.
            # Similar to thermo parameter in LAMMPS.
            'thermo_output': [ 
                'custom', 'step', 'time', 'spcpu',
                'temp', 'press',
                'pe', 'ke',
                'enthalpy', 'evdwl', 'ecoul', 'epair',
                'ebond', 'eangle', 'edihed',
                'elong', 'etail', 'emol',
                'ecouple', 'econserve', 'etotal',
                'lx', 'ly', 'lz',
            ]
        }

        print(f"could not find {file_name}. defualt params would be used to initialization and default params will be saved to params.json file.")
        with open('params.json', 'w', encoding = 'utf-8') as outfile:
            json.dump(params, outfile, indent = 4)

        with open('params.json', 'r', encoding = 'utf-8') as infile:
            default_params = json.load(infile)

    return default_params
