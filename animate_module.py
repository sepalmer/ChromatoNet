import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as clt
from matplotlib import animation
from matplotlib import colors

'''
Animates the four .csv files produced by squid_sim. Just change the name_scheme to whatever you put as 'name'
in the CHR.run() function of squid_sim. So for example, if I have CHR.run(100000, name='3pop'), I'll get 
.csv files named 'x_pos_3pop.csv', etc, and would put name_scheme = '3pop' here.

Dependencies: 
requires working version of ffmpeg on machine path to save. If you just want to see the animation, this line can be commented out.

'''

def animate(name_scheme, speedup=10):

    speedup = 10 # factor by which to speed up the sim
    
    try:
        time = np.loadtxt('time_'+name_scheme+'.csv', delimiter=' ')
        tstep = 1000*(time[1]-time[0])/speedup
        print('One sim realtime duration:', max(time), 'seconds')
        print('Sped up by factor of:', speedup)
    except:
        print("Could not load time data, using default timestep")
        tstep = 50  # Default 50ms between frames

    # Load data with error handling for different formats
    try:
        # First check what's actually in the file
        with open('x_pos_'+name_scheme+'.csv', 'r') as f:
            lines = f.readlines()
            print(f"CSV file has {len(lines)} lines")
            print(f"First line: {lines[0][:100]}...")  # Show first 100 chars
            if len(lines) > 1:
                print(f"Second line: {lines[1][:100]}...")
        
        x = np.loadtxt('x_pos_'+name_scheme+'.csv', delimiter=' ')
        print(f"Successfully loaded with shape: {x.shape}")
    except Exception as e:
        print(f"Error loading with space delimiter: {e}")
        try:
            x = np.loadtxt('x_pos_'+name_scheme+'.csv', delimiter=',')
            print(f"Successfully loaded with comma delimiter, shape: {x.shape}")
        except Exception as e2:
            print(f"Could not load x_pos_{name_scheme}.csv with any delimiter: {e2}")
            return None
            
    try:
        p = np.loadtxt('pops_'+name_scheme+'.csv').astype(int)
    except:
        print("Could not load pops data, using default")
        # Estimate number of chromatophores and use default population
        n_chromats = x.shape[1] // 3 if x.ndim > 1 else len(x) // 3
        p = np.ones(n_chromats, dtype=int)
    
    # Handle different data shapes
    print(f"Loaded x data shape: {x.shape}")
    print(f"x data type: {type(x)}")
    
    # Check if np.savetxt saved it as a 1D array when it should be 2D
    if x.ndim == 1:
        # Try to determine how many timesteps there should be
        try:
            time_data = np.loadtxt('time_'+name_scheme+'.csv', delimiter=' ')
            expected_timesteps = len(time_data)
            expected_cols = len(x) // expected_timesteps
            
            if len(x) % expected_timesteps == 0:
                print(f"Reshaping from {x.shape} to ({expected_timesteps}, {expected_cols})")
                x = x.reshape(expected_timesteps, expected_cols)
            else:
                print("Warning: Cannot determine proper reshape, creating static animation")
                x = x.reshape(1, -1)
        except:
            print("Warning: Only one timestep found, creating static animation")
            x = x.reshape(1, -1)
    
    # Check if we have valid data
    if x.shape[1] < 3:
        print(f"Error: Not enough data points: {x.shape[1]}. Need at least 3.")
        return None
        
    # Ensure data length is divisible by 3 (x, y, radius for each chromatophore)
    n_points = x.shape[1] // 3
    if x.shape[1] % 3 != 0:
        print(f"Warning: Data length {x.shape[1]} not divisible by 3. Truncating to {n_points*3}.")
        x = x[:, :n_points*3]
    
    xrs = x.reshape((len(x), int(len(x[0])/3), 3))
    print(f"Reshaped to: {xrs.shape}")
    
    # Use nanmin/nanmax to handle any NaN values
    xl = np.nanmin(xrs[:,:,0])-1
    xh = np.nanmax(xrs[:,:,0])+1
    yl = np.nanmin(xrs[:,:,1])-1
    yh = np.nanmax(xrs[:,:,1])+1

    popcols = ['red', 'yellow', 'brown']
    ccmap = [popcols[i-1] for i in p[:n_points]]  # Ensure we don't exceed array bounds

    fig = plt.figure(figsize=(10,10*(yh-yl)/(xh-xl)))
    ax = plt.axes(xlim = (xl, xh), ylim = (yl, yh))
    plt.axis('off')

    patches = []
    for c in xrs[0]:
        patch = plt.Circle((c[0],c[1]), c[2] + 0.05)
        patches.append(patch)

    collection = clt.PatchCollection(patches, facecolors=ccmap)

    def init():
        ax.add_collection(collection)
        return collection,

    def animate_frame(i):
        patches = []
        for c in xrs[i]:
            patch = plt.Circle((c[0],c[1]), c[2] + 0.05)
            patches.append(patch)
        collection.set_paths(patches)
        return collection,

    anim = animation.FuncAnimation(fig, animate_frame,
                                    init_func = init,
                                    frames = len(xrs), 
                                    interval = tstep, # time between saved measurements in ms
                                    blit = True)

    return(anim)