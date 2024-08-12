import drprobe as drp
import numpy as np
from io import StringIO
import math
import os

# Define the path to the file containing the initial parameters
para_path = "Test_results/TEMfitting_tests/example/para_results/initial_paras.txt"

# Initialize an empty list to store cleaned lines
cleaned_lines = []

def count_files_in_directory(directory_path):
    # List all items in the directory
    items = os.listdir(directory_path)
    # Filter out directories, keeping only files
    files = [item for item in items if os.path.isfile(os.path.join(directory_path, item))]
    # Return the number of files
    return len(files)

# Open the file in read mode
with open(para_path, 'r') as file:
    # Iterate over each line in the file
    for line in file:
        # Strip whitespace and remove square brackets from the line
        cleaned_line = line.strip().replace('[', '').replace(']', '')
        # Append the cleaned line to the list
        cleaned_lines.append(cleaned_line)

# Join the cleaned lines into a single string with newline characters
cleaned_data = '\n'.join(cleaned_lines)

# Load the cleaned data into a numpy array
data = np.loadtxt(StringIO(cleaned_data))

# Define the path to the folder where simulation resulsts will be saved
folder_save_file = 'Test_results/TEMfitting_tests/example/results_img/t22/task/img'
# Note: para_path and folder_save_file should be the same as the path in input.txt！！！

# simulation 
a, b, c = np.genfromtxt('Test_results/TEMfitting_tests/example/target/STO110.cel', skip_header=1, skip_footer=1, usecols=(1, 2, 3))[0]
nx, ny = 128, 90
nz = 4    

# Iterate over each item in 'data' with its index.
for folder_num, x in enumerate(data):
    
    init_num = count_files_in_directory(folder_save_file)
    fr_num = init_num + folder_num
    
    # Note: 'Num_x' is a list of the elements from 'x' after converting from numpy scalar types.
    Num_x = [i.item() for i in x]  
    
    # Extract the elements from the Num_x list and assign it to simulation parameters.
    abf = Num_x[0]
    tilt_x = -0.362
    tilt_y = 0.213      
    vibration = 0.02
    vibration_matrix = (1,vibration,vibration,0)
    defocus = Num_x[1] #C1
    thick  = 43
    ht = 300
    wlkev = 1.2398419843320025  # c * h / q_e * 1E6
    twomc2 = 1021.9978999923284  # 2*m0*c**2 / q_e     
    STF_HT2WL = wlkev / math.sqrt(ht * (twomc2 + ht))

    aberrations_dict = {0:(-0.03307108882342791,0.005962697858410755), 1: (defocus, 0), 2:(-0.4198138840789767,0.2559249505403436), 
                           3:(-50,-32.01675073563152), 4:(32.32397201121741,-11.384340575567556), 5:(-15000, 0),11:(1000000,0)}
        
    wave_output = 'Test_results/TEMfitting_tests/example/results_img/t22/task/wav/YAP.wav'
    wavimg_prm = 'Test_results/TEMfitting_tests/example/results_img/t22/task/wavimg.prm'
    slice_name = 'Test_results/TEMfitting_tests/example/results_img/t22/task/slc/YAP'
    msa_prm = 'Test_results/TEMfitting_tests/example/results_img/t22/task/msa.prm'
    drp.commands.celslc(cel_file='Test_results/TEMfitting_tests/example/target/STO_new.cel', # location of cel file
                    slice_name=slice_name, # target file names
                    nx=nx,        # number of sampling points along x
                    ny=ny,        # number of sampling points along y
                    nz=nz,         # number of sampling points along z
                    ht=ht,       # high tension
                    abf=abf,
                    absorb=False,  # apply absorptive form factors
                    dwf=True,     # apply Debye-Waller factors
                    output=True,  # Command line output (prints executed command)
                    pot=True      # Saves potentials
                   )
            
    msa = drp.msaprm.MsaPrm()
    msa. conv_semi_angle = 0.2
    msa.wavelength = STF_HT2WL  # wavelength (in nm) for 300 keV electrons
    msa.tilt_x = tilt_x # object tilt along x in degree
    msa.tilt_y = tilt_y # object tilt along y in degree
    msa.slice_files = slice_name # location of phase gratings
    msa.number_of_slices = 4 # Number of slices in phase gratings
    msa.det_readout_period = 1 # Readout after every 2 slices
    msa.tot_number_of_slices = 52 # Corresponds to 5 layers / unit cells of SrTiO3
            # Save prm file
    msa.save_msa_prm(msa_prm)
        
    drp.commands.msa(prm_file = msa_prm, 
                 output_file= wave_output,
                 ctem=True, # Flag for conventional TEM simulation (otherwise calculates STEM images)
                 output=True)
    wavimg = drp.WavimgPrm()
    sl = round(thick) # Perform simulation for slice # 2       
    wave_files = 'Test_results/TEMfitting_tests/example/results_img/t22/task/wav/YAP_sl{:03d}.wav'.format(sl)
    output_files = 'Test_results/TEMfitting_tests/example/results_img/t22/task/img/YAP_sl{:03d}_{}.dat'.format(sl, fr_num)
    stl = '{:03d}'.format(sl)
    fake_data = 'Test_results/TEMfitting_tests/example/results_img/t22/task/img/'+'YAP_sl'+stl+ '_' +str(fr_num) +'.dat'
    # Setup (fundamental) MSA parameters
    wavimg.high_tension = ht
    wavimg.mtf = (1, 2.06, 'Test_results/TEMfitting_tests/example/target/MTF-US2k-300.mtf')  
    wavimg.oa_radius = 250 # Apply objective aperture [mrad]
    wavimg.oa_position = (0, 0)
    # wavimg.vibration = vibration_matrix # Apply isotropic image spread of 16 pm rms displacement
    wavimg.vibration = vibration_matrix
    wavimg.spat_coherence = (1, 0.2)
    wavimg.temp_coherence = (1, 3) # Apply spatial coherence
    wavimg.wave_sampling = (a / nx, b / ny)
    wavimg.wave_files = wave_files          
    wavimg.wave_dim = (nx, ny)
    wavimg.aberrations_dict = aberrations_dict # apply aberrations
    wavimg.output_files = output_files # Simulate for slice number "sl"
    wavimg.output_format = 0
    wavimg.flag_spec_frame = 1
    wavimg.output_dim = (nx, ny)
    wavimg.output_sampling = (a / nx, b / ny)

    # Save wavimg Parameter file
    wavimg.save_wavimg_prm(wavimg_prm)
    drp.commands.wavimg(wavimg_prm, output=True)
    