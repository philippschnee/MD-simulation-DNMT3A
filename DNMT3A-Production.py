from __future__ import print_function
from openmm.app import *
from openmm import *
import openmm as mm
from openmm.unit import *
from sys import stdout
from pdbfixer import PDBFixer
from mdtraj.reporters import HDF5Reporter
import mdtraj as mdt
import os
import re
import numpy as np

# input parameters. Needed to find and name the files.

Variant = 'WT'          # WT: Wild Type or MT: Mutant.
sim_time = '25ns'       # simulation time of each replicate.
number_replicates = 10  # the number of replicates you want to simulate

# parameters to definde your system

dt = 0.001*picoseconds      # this is the stepsize (1fs)
temperature = 300*kelvin
friction = 1/picosecond
sim_ph = 7.0

# create folder where all simulation outputs are stored

traj_folder = 'Trajectory_DNMT3A-{}-{}'.format(Variant, sim_time)
os.system('mkdir {0}'.format(traj_folder))


count = 1
while (count <= number_replicates):
    
    # input Files
    pdb = PDBFile('DNMT3A-{}.pdb'.format(Variant))          # this is your starting strcture of the protein and peptide
    protein = app.Modeller(pdb.topology, pdb.positions)
    sim_forcefield = ('amber14-all.xml')                    # this is the used forcefield
    sim_watermodel = ('amber14/tip4pew.xml')                # this is the used water model
    sim_gaff = 'gaff.xml'
    
    ligand_names = ['SMA', 'SMB']                           # these are the two SAM molecules in the system: SMA-> SAM-1, SMB-> SAM-2
    ligand_xml_files = ['SMA.xml', 'SMB.xml']               # these are the parameter files for the SAM molecules as they are not included in the standart forcefield                    
    ligand_pdb_files = ['SMA.pdb', 'SMB.pdb']               # these are the PDB files of the two SAM molecules                                         
    
    # simulation Options
    Simulate_Steps = 25000000   # 25 ns, simulation time of the main simualtion                 
    
    npt_eq_Steps = 3000000      # 3 ns, simulation time of the NPT equilibration
    restr_eq_Steps = 3000000    # 3 ns, simulation time of the restrained equilibration
    free_eq_Steps = 3000000     # 3 ns, simulation time of the unrestrained equilibration
    
    restrained_eq_atoms = 'protein and name CA or chainid 4 or chainid 5'  # here we restrain the protein during the NPT equilibration phase
    force_eq_atoms = 2                                                     # kilojoules_per_mole/unit.angstroms
    
    restrained_eq_atoms2 = 'chainid 6 or chainid 7'                        # here we can restrain the two SAM molecules
    force_eq_atoms2 = 1                                                    # kilojoules_per_mole/unit.angstroms
    
    restrained_ligands = False # set this to true or false, depending if you want to restain molecules                     
    
    # information for the hardware you are simulating on. These parameters are deigned for NVIDIA GPUs
    platform = Platform.getPlatformByName('CUDA')
    gpu_index = '0'                                    # GPUs can be enumerated (0, 1, 2), if you don't know how this is set up, stick to 0
    platformProperties = {'Precision': 'single','DeviceIndex': gpu_index}
    trajectory_out_atoms = 'protein or chainid 4 or chainid 5 or resname SMA or resname SMB' # these are the atoms being saved in the trajectory, currently DNMT3A, the DNA and the two SAM molecules, waters and solvent iions are excluded
    trajectory_out_interval = 10000 # frames are wirtten into the trajectory every 10000 calcualted steps. For higher resolution decrease this. However file size and calculation times increases
    
    # every residue is protonated according to the pH except for the residues mentioned below
    protonation_dict = {} #only for manual protonation
    
    # build force field object | all xml parameter files are bundled here
    xml_list = [sim_forcefield, sim_gaff, sim_watermodel]
    for lig_xml_file in ligand_xml_files:
        xml_list.append(lig_xml_file)
    forcefield = app.ForceField(*xml_list)
    
    # protonate protein and also assign custom protonations defined above
    protonation_list = []
    key_list=[]
    if len(protonation_dict.keys()) > 0:
        for chain in protein.topology.chains():
            chain_id = chain.id
            protonations_in_chain_dict = {}
            for protonation_tuple in protonation_dict:
                if chain_id == protonation_tuple[0]:
                    residue_number = protonation_tuple[1]
                    protonations_in_chain_dict[int(residue_number)] = protonation_dict[protonation_tuple]
                    key_list.append(int(residue_number)) 
    if len(protonation_dict.keys()) > 0:
        protein.addHydrogens(forcefield, pH = sim_ph, variants = protonation_list)
    else:
        protein.addHydrogens(forcefield, pH = sim_ph)
    
    # add ligand structures to the model
    for lig_pdb_file in ligand_pdb_files:
        ligand_pdb = app.PDBFile(lig_pdb_file)
        protein.add(ligand_pdb.topology, ligand_pdb.positions)
    
    # Generation and Solvation of Box
    print('Generation and Solvation of Box')
    boxtype = 'rectangular' # cubic or rectangular
    box_padding = 1.0 # in nanometers. This defines the distance of the protein to the box edges
    x_list = []
    y_list = []
    z_list = []
    
    # get atom indices for protein plus ligands
    for index in range(len(protein.positions)):
        x_list.append(protein.positions[index][0]._value)
        y_list.append(protein.positions[index][1]._value)
        z_list.append(protein.positions[index][2]._value)
    x_span = (max(x_list) - min(x_list))
    y_span = (max(y_list) - min(y_list))
    z_span = (max(z_list) - min(z_list))
    
    # build box and add solvent
    d =  max(x_span, y_span, z_span) + (2 * box_padding)
    d_x = x_span + (2 * box_padding)
    d_y = y_span + (2 * box_padding)
    d_z = z_span + (2 * box_padding)
    
    prot_x_mid = min(x_list) + (0.5 * x_span)
    prot_y_mid = min(y_list) + (0.5 * y_span)
    prot_z_mid = min(z_list) + (0.5 * z_span)
    
    box_x_mid = d_x * 0.5
    box_y_mid = d_y * 0.5
    box_z_mid = d_z * 0.5
    
    shift_x = box_x_mid - prot_x_mid
    shift_y = box_y_mid - prot_y_mid
    shift_z = box_z_mid - prot_z_mid
    
    solvated_protein = app.Modeller(protein.topology, protein.positions)
    
    # shift coordinates to the middle of the box
    for index in range(len(solvated_protein.positions)):
        solvated_protein.positions[index] = (solvated_protein.positions[index][0]._value + shift_x, solvated_protein.positions[index][1]._value + shift_y, solvated_protein.positions[index][2]._value + shift_z)*nanometers
    
    # add box vectors and solvate
    if boxtype == 'cubic':
        solvated_protein.addSolvent(forcefield, model='tip4pew', neutralize=True, ionicStrength=0.1*molar, boxVectors=(mm.Vec3(d, 0., 0.), mm.Vec3(0., d, 0.), mm.Vec3(0, 0, d)))
    elif boxtype == 'rectangular':
        solvated_protein.addSolvent(forcefield, model='tip4pew', neutralize=True, ionicStrength=0.1*molar, boxVectors=(mm.Vec3(d_x, 0., 0.), mm.Vec3(0., d_y, 0.), mm.Vec3(0, 0, d_z)))
    
    
    # Building System
    
    print('Building system...')
    topology = solvated_protein.topology
    positions = solvated_protein.positions
    selection_reference_topology = mdt.Topology().from_openmm(solvated_protein.topology)
    trajectory_out_indices = selection_reference_topology.select(trajectory_out_atoms)
    restrained_eq_indices = selection_reference_topology.select(restrained_eq_atoms)
    restrained_eq_indices2 = selection_reference_topology.select(restrained_eq_atoms2)
    system = forcefield.createSystem(topology, nonbondedMethod=app.PME, nonbondedCutoff=1.0*nanometers,ewaldErrorTolerance=0.0005, constraints=HBonds, rigidWater=True)
    integrator = LangevinIntegrator(temperature, friction, dt)
    simulation = Simulation(topology, system, integrator, platform, platformProperties)
    simulation.context.setPositions(positions)
    
    
    # Minimize
    
    print('Performing energy minimization...')
    print(simulation.context.getState(getEnergy=True).getPotentialEnergy())
    app.PDBFile.writeFile(simulation.topology, positions, open(traj_folder + '/' + 'PreMin.pdb', 'w'), keepIds=True) # this writes a PDB files of the system before the energy minimization
    simulation.minimizeEnergy() # here the energy of the system is minimized
    min_pos = simulation.context.getState(getPositions=True).getPositions()
    app.PDBFile.writeFile(simulation.topology, positions, open(traj_folder + '/' + 'PostMin.pdb', 'w'), keepIds=True) # this writes a PDB files of the system after the energy minimization
    print(simulation.context.getState(getEnergy=True).getPotentialEnergy())
    print('System is now minimized')
    
    
    # adding the restraints to the system
    
    force = mm.CustomExternalForce("(k/2)*periodicdistance(x, y, z, x0, y0, z0)^2")
    force.addGlobalParameter("k", force_eq_atoms*kilojoules_per_mole/angstroms**2)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")
    
    force2 = mm.CustomExternalForce("(k/2)*periodicdistance(x, y, z, x0, y0, z0)^2")
    force2.addGlobalParameter("k", force_eq_atoms2*kilojoules_per_mole/angstroms**2)
    force2.addPerParticleParameter("x0")
    force2.addPerParticleParameter("y0")
    force2.addPerParticleParameter("z0")
    
    if restrained_ligands:
        for res_atom_index in restrained_eq_indices2:
            force2.addParticle(int(res_atom_index), min_pos[int(res_atom_index)].value_in_unit(nanometers))
        system.addForce(force2)
    
    for res_atom_index in restrained_eq_indices:
        force.addParticle(int(res_atom_index), min_pos[int(res_atom_index)].value_in_unit(nanometers))
    system.addForce(force)
    
    # NPT Equilibration
    
    # add barostat for NPT
    system.addForce(mm.MonteCarloBarostat(1*atmospheres, temperature, 25))
    simulation.context.setPositions(min_pos)
    simulation.context.setVelocitiesToTemperature(temperature*kelvin)
    simulation.reporters.append(app.StateDataReporter(stdout, 10000, step=True, potentialEnergy=True, temperature=True, progress=True, remainingTime=True, speed=True, totalSteps=npt_eq_Steps, separator='\t'))
    simulation.reporters.append(HDF5Reporter(traj_folder + '/' + 'EQ_NPT.h5', 10000, atomSubset=trajectory_out_indices))
    print('restrained NPT equilibration...')
    simulation.step(npt_eq_Steps)
    
    forces_record = simulation.context.getState(getForces=True).getForces() # here the forces are recorded in order to spot large peaks, indicating a fragile system
    for i, f in enumerate(forces_record):
        if norm(f) > 1e4*kilojoules_per_mole/nanometer:
            print(i, f)
    
    state_npt_EQ = simulation.context.getState(getPositions=True, getVelocities=True)
    positions = state_npt_EQ.getPositions()
    app.PDBFile.writeFile(simulation.topology, positions, open(traj_folder + '/' + 'post_NPT_EQ.pdb', 'w'), keepIds=True)
    print(simulation.context.getState(getEnergy=True).getPotentialEnergy())
    print('Successful NPT equilibration!')
    
    
    # Free Equilibration
    # forces: 0->HarmonicBondForce, 1->HarmonicAngleForce, 2->PeriodicTorsionForce, 3->NonbondedForce, 4->CMMotionRemover, 5->CustomExternalForce, 6->CustomExternalForce, 7->MonteCarloBarostat
    n_forces = len(system.getForces())
    system.removeForce(n_forces-2)
    print('force removed')
    
    # optional ligand restraint to force slight conformational changes
    if restrained_ligands:
        integrator = mm.LangevinIntegrator(temperature*kelvin, 1/picosecond, 0.001*picoseconds)
        simulation = app.Simulation(solvated_protein.topology, system, integrator, platform, platformProperties)
        simulation.context.setState(state_npt_EQ)
        simulation.reporters.append(app.StateDataReporter(stdout, 10000, step=True, potentialEnergy=True, temperature=True, progress=True, remainingTime=True, speed=True, totalSteps=restr_eq_Steps, separator='\t'))
        simulation.reporters.append(HDF5Reporter(traj_folder + '/' + 'free_BB_restrained_NPT_EQ.h5', 10000, atomSubset=trajectory_out_indices))
        print('free BB NPT equilibration of protein with restrained SAM...')
        simulation.step(restr_eq_Steps)
        state_free_EQP = simulation.context.getState(getPositions=True, getVelocities=True)
        positions = state_free_EQP.getPositions()
        app.PDBFile.writeFile(simulation.topology, positions, open(traj_folder + '/' + 'free_BB_restrained_NPT_EQ.pdb', 'w'), keepIds=True)
        print('Successful free BB, SAM restrained equilibration!')
    
        # equilibration with free ligand   
        n_forces = len(system.getForces())
        system.removeForce(n_forces-2)
        integrator = mm.LangevinIntegrator(temperature*kelvin, 1/picosecond, 0.001*picoseconds)
        simulation = app.Simulation(solvated_protein.topology, system, integrator, platform, platformProperties)
        simulation.context.setState(state_free_EQP)
        simulation.reporters.append(app.StateDataReporter(stdout, 10000, step=True, potentialEnergy=True, temperature=True, progress=True, remainingTime=True, speed=True, totalSteps=free_eq_Steps, separator='\t'))
        simulation.reporters.append(HDF5Reporter(traj_folder + '/' + 'free_NPT_EQ.h5', 10000, atomSubset=trajectory_out_indices))
        print('SAM free NPT equilibration...')
        simulation.step(free_eq_Steps)
        state_free_EQ = simulation.context.getState(getPositions=True, getVelocities=True)
        positions = state_free_EQ.getPositions()
        app.PDBFile.writeFile(simulation.topology, positions, open(traj_folder + '/' + 'free_NPT_EQ.pdb', 'w'), keepIds=True)
        print('Successful SAM free equilibration!')
      
    else:
        # remove ligand restraints for free equilibration (remove the second last force object, as the last one was the barostat)
        n_forces = len(system.getForces())
        system.removeForce(n_forces-2)
        integrator = mm.LangevinIntegrator(temperature, 1/picosecond, 0.001*picoseconds)
        simulation = app.Simulation(solvated_protein.topology, system, integrator, platform, platformProperties)
        simulation.context.setState(state_npt_EQ)
        simulation.reporters.append(app.StateDataReporter(stdout, 10000, step=True, potentialEnergy=True, temperature=True, progress=True, remainingTime=True, speed=True, totalSteps=free_eq_Steps, separator='\t'))
        simulation.reporters.append(HDF5Reporter(traj_folder + '/' + 'EQ_NPT_free.h5', 10000, atomSubset=trajectory_out_indices))
        print('free NPT equilibration...')
        simulation.step(free_eq_Steps)
        state_free_EQ = simulation.context.getState(getPositions=True, getVelocities=True)
        positions = state_free_EQ.getPositions()
        app.PDBFile.writeFile(simulation.topology, positions, open(traj_folder + '/' + 'free_NPT_EQ.pdb', 'w'), keepIds=True)
        print('Successful free equilibration!')

    # Simulate
    print('Simulating...')
    
    # create new simulation object for production run with new integrator
    integrator = mm.LangevinIntegrator(temperature, 1/picosecond, 0.001*picoseconds)
    simulation = app.Simulation(solvated_protein.topology, system, integrator, platform, platformProperties)
    simulation.context.setState(state_free_EQ)
    simulation.reporters.append(app.StateDataReporter(stdout, 10000, step=True, potentialEnergy=True, temperature=True, progress=True, remainingTime=True, speed=True, totalSteps=Simulate_Steps, separator='\t'))
    simulation.reporters.append(HDF5Reporter(traj_folder + '/' + 'production_DNMT3A-{}_{}_Replicate{}.h5'.format(Variant, sim_time, count), 10000, atomSubset=trajectory_out_indices))
    print('production run of replicate {}...'.format(count))
    simulation.step(Simulate_Steps)
    state_production = simulation.context.getState(getPositions=True, getVelocities=True)
    state_production = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
    final_pos = state_production.getPositions()
    app.PDBFile.writeFile(simulation.topology, final_pos, open(traj_folder + '/' + 'production_DNMT3A-{}_{}_Replicate{}.pdb'.format(Variant, sim_time, count), 'w'), keepIds=True)
    print('Successful production of replicate {}...'.format(count))
    del(simulation)
 
    count = count+1