import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rcParams, cm

def rotmat_x(th):

    # Input: 
    # th = desired (anticlockwise) rotation around x
    # Output: 
    # Rx = the desired rotation matrix
    
    Rx = np.array([[         1,          0,          0],
                   [         0, np.cos(th),-np.sin(th)],
                   [         0, np.sin(th), np.cos(th)]])
    return Rx

def rotmat_y(th):    

    # Input: 
    # th = desired (anticlockwise) rotation around y
    # Output: 
    # Ry = the desired rotation matrix
    
    Ry = np.array([[ np.cos(th),       0, np.sin(th)],
                   [          0,       1,          0],
                   [-np.sin(th),       0, np.cos(th)]])
    return Ry

def rotmat_z(th):
   
    # Input: 
    # th = desired (anticlockwise) rotation around z
    # Output: 
    # Rz = the desired rotation matrix

    Rz = np.array([[np.cos(th),-np.sin(th),    0],
                   [np.sin(th), np.cos(th),    0],
                   [         0,          0,    1]])
    return Rz

def rotmat_vecspace(B):
    
    # Input: 
    # B = the vector describing the B field
    # Output: 
    # Rv = the desired rotation matrix to orientate B along +z

    #  find angle of B from x axis in x-y plane    
    th_x = np.arctan2(B[1][0],B[0][0])
    #  find angle of B from z axis    
    th_z = np.arctan2(np.linalg.norm(B[0:2]),B[2][0])
    #  Rotate B around z axis, and then around y axis, to +z direction
    Rv = rotmat_y(-th_z) @ rotmat_z(-th_x);
    return Rv

def rotmat_bfield(B,t,gamma_bar):
   
    # Input: 
    # B = the vector describing the B field (in Tesla)
    # t = the duration of the B field (in s)
    # Output: 
    # Rb = the desired rotation matrix to rotate M around B

    # what angle do we need to rotate?
    z_th = 2 * np.pi * gamma_bar * np.linalg.norm(B) * t 
    Rb = rotmat_z(-z_th)
    return Rb
    
    
def relax(Mi,T1,T2,t):
   
    # Input: 
    # Mi = the input magnetisation vector ([Mx, My, Mz].')
    #      (assuming equilib. magn. M0 = [0,0,1].')
    # T1 = the T1 of our tissue (in s) 
    # T2 = the T2 of our tissue (in s) 
    # t  = the duration of the relaxation process (in s) 
    # Output: 
    # Mo = the magnetisation vector after undergoing relaxation ([Mx, My, Mz].')

    
    # Let's create our two relaxation operators
    R_op1 = np.array([[  np.exp(-t/T2),            0 ,             0],
                      [              0, np.exp(-t/T2),             0],
                      [              0,             0, np.exp(-t/T1)]])
    
    R_op2 = np.array([[              0],
                      [              0], 
                      [1-np.exp(-t/T1)]]) 
    
    Mo = R_op1 @ Mi + R_op2;
    
    return Mo

def bloch_sim_events(M,events,T1,T2,dB0,gamma_bar):
    
    # Input: 
    # M = the input magnetisation vector ([Mx, My, Mz].')
    # events = 2D pulse sequence event array of size #blocks x 3 (2nd dim: duration, |B1|, B1 phase)
    # T1 = the T1 of our tissue (in s) 
    # T2 = the T2 of our tissue (in s) 
    # dB0 = range of delta B0 to cover all isochromats (in Tesla)
    # Output: 
    # M = the magnetisation vector after pulse sequence events ([Mx, My, Mz].')    
    
    events = events.reshape(-1,3)
    for t in range(events.shape[0]):
        t_res = events[t,0]   # read in duration of block
        B1 = events[t,1]      # read in B1 magnitude
        phi = events[t,2]     # read in B1 phase
        ## RF PULSE
        B_rf=np.array([B1*np.sin(phi),-B1*np.cos(phi),0]).reshape(3,1)
        # Loop over each M component described by the different B0 values, and apply the Bloch eqs. for rotation
        for dB0_idx in range(np.size(dB0)):
            # What B field does this M component experience?
            B_mod = B_rf + np.array([0,0,dB0[dB0_idx]]).reshape(3,1)
            # Apply corresponding rotation
            M[:,dB0_idx] = np.transpose(rotmat_vecspace(B_mod)) @ (rotmat_bfield(B_mod,t_res,gamma_bar) @ (rotmat_vecspace(B_mod) @ M[:,dB0_idx].flatten()))
            # Apply relaxation over this time frame
            M[:,dB0_idx] = relax(M[:,dB0_idx].reshape(3,1),T1,T2,t_res).flatten()

    return M


### FROM HERE ON THESE ARE PLOTTING FUNCTIONS, SO LESS INTERESTING!

def setup_M_plots(M,dB0):
    
    fig = plt.figure(figsize=(12,5))
    fig.set_facecolor('#DAE3F3')

    ax1 = fig.add_subplot(121, projection='3d',computed_zorder=False)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(224)
    fig.subplots_adjust(wspace=0.2,hspace=0.4)
    tt = fig.suptitle('')

    M_padded = np.zeros([3,2,np.size(M,1)])
    M_padded[:,1,:]=M
    Mpc1, = ax1.plot3D (M[0,:],M[1,:],M[2,:],'k', lw=2, zorder=2)
    Mpc2, = ax2.plot (M[0,:],M[1,:],'k', lw=2, zorder=2)
    Mxy1, = ax3.plot (dB0,np.abs(M[0,:]+1j*M[1,:]),'k', lw=2, zorder=2)


    for ln_no in range(np.size(dB0)):
        Mp1, = ax1.plot3D (M_padded[0,:,ln_no],M_padded[1,:,ln_no],M_padded[2,:,ln_no], c=cm.rainbow((ln_no+0.01)/np.size(dB0)), lw=0.4, zorder=3)
        Mp2, = ax2.plot (M_padded[0,:,ln_no],M_padded[1,:,ln_no], c=cm.rainbow((ln_no+0.01)/np.size(dB0)), lw=0.4, zorder=3)
        Mxy2, = ax3.plot (np.array([dB0[ln_no],dB0[ln_no]]),np.array([0,np.abs(M[0,ln_no]+1j*M[1,ln_no])]),c=cm.rainbow((ln_no+0.01)/np.size(dB0)), zorder=1)

    M_RF, = ax1.plot3D (np.array([0,0]),np.array([0,0]),np.array([0,0]),'k',lw=8)
    M_RF2, = ax2.plot (np.array([0,0]),np.array([0,0]),'k',lw=8)
        
    ax_scl = max(abs(np.linalg.norm(M,axis=0)))
    # set up the subplots as needed
    ax1.set_xlim((-ax_scl, ax_scl))            
    ax1.set_ylim((-ax_scl, ax_scl))
    ax1.set_zlim((-ax_scl, ax_scl))
    ax1.grid(False)
    ax1.set_xticks([0])
    ax1.set_yticks([0]) 
    ax1.set_zticks([0]) 
    ax1.set_facecolor('#DAE3F3')
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    ax1.set_zlabel('')
    ax1.xaxis.set_ticklabels('X')
    ax1.yaxis.set_ticklabels('Y')
    ax1.zaxis.set_ticklabels('Z')
    ax1.plot(0.8*ax_scl*np.sin(np.linspace(0,2*np.pi,100)),0.8*ax_scl*np.cos(np.linspace(0,2*np.pi,100)),np.linspace(0,0,100),'--k', lw=1, zorder=1)
    ax1.plot(ax_scl*np.array([-1,1]),np.array([0,0]),np.array([0,0]),'k', lw=1, zorder=1)
    ax1.plot(np.array([0,0]),ax_scl*np.array([-1,1]),np.array([0,0]),'k', lw=1, zorder=1)
    ax1.plot(np.array([0,0]),np.array([0,0]),ax_scl*np.array([-1,1]),'k', lw=1, zorder=1)
    ax1.set_title('a) $M_{XYZ}$ across isochromats')

    ax2.set_xlim((-ax_scl, ax_scl))            
    ax2.set_ylim((-ax_scl, ax_scl))
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.yaxis.set_label_position("right")
    ax2.set_aspect('equal')
    ax2.set_xticks([])
    ax2.set_yticks([]) 
    ax2.plot(0.8*ax_scl*np.sin(np.linspace(0,2*np.pi,100)),0.8*ax_scl*np.cos(np.linspace(0,2*np.pi,100)),'--k', lw=1, zorder=1)
    ax2.plot(ax_scl*np.array([-1,1]),np.array([0,0]),'k', lw=1, zorder=1)
    ax2.plot(np.array([0,0]),ax_scl*np.array([-1,1]),'k', lw=1, zorder=1)
    ax2.set_title('b) $M_{XY}$ across isochromats')

    #M_plot, = ax3.plot3D (M_padded[0,:],M_padded[1,:],M_padded[2,:],'r', lw=0.2)

    ax3.set_ylim(0, ax_scl)
    ax3.set_xlim(dB0[0], dB0[-1])
    ax3.set_box_aspect(1)
    ax3.set_xticks([dB0[0],0,-dB0[0]]) 
    ax3.xaxis.set_ticklabels(['-1/TR','0','1/TR'])
    ax3.set_xlabel('Off-resonance')
    ax3.set_ylabel('$|M_{XY}|$')
    ax3.set_title('c) $|M_{XY}|$ across isochromats')

    txt_RF = ax1.text(0,0,0, "", color='k', weight='bold')
    txt_RF2 = ax2.text(0,0, "", color='k', weight='bold')


    return fig,ax1,ax2,ax3,tt,Mpc1,Mpc2,Mxy1

def update_M_plots(dB0,M,ax1,ax2,ax3,Mpc1,Mpc2,Mxy1):
    
    for dB0_idx in range(np.size(dB0)):
        ax_temp = ax1.get_lines()[dB0_idx+1]
        ax_temp.set_data_3d(np.array([0,M[0,dB0_idx]]),np.array([0,M[1,dB0_idx]]),np.array([0,M[2,dB0_idx]]))
        ax_temp = ax2.get_lines()[dB0_idx+1]
        ax_temp.set_data(np.array([0,M[0,dB0_idx]]),np.array([0,M[1,dB0_idx]]))
        ax_temp = ax3.get_lines()[dB0_idx+1]
        ax_temp.set_data(np.array([dB0[dB0_idx],dB0[dB0_idx]]),[0,np.abs(M[0,dB0_idx]+1j*M[1,dB0_idx])])
    Mpc1.set_data_3d(M[0,:],M[1,:],M[2,:])
    
    Mpc2.set_data(M[0,:],M[1,:])
    Mxy1.set_data(dB0,np.abs(M[0,:]+1j*M[1,:]))
    
    

def setup_Fstate_plots(M,dB0):
    
    fig = plt.figure(figsize=(12,5))
    fig.set_facecolor('#DAE3F3')

    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    fig.subplots_adjust(wspace=0.2,hspace=0.4)
    tt = fig.suptitle('')

    F = np.fft.ifftshift(np.fft.fft(np.fft.fftshift(1j*M[0,:]+M[1,:])))/len(dB0)
    

    ax2.stem (np.arange(0,len(F)/2)-(len(F)/4),np.abs(F[1::2]),'k', markerfmt=" ",basefmt=" ")

    M_padded = np.zeros([3,2,np.size(M,1)])
    M_padded[:,1,:]=M

    for ln_no in range(np.size(dB0)):
        Mp2, = ax1.plot (M_padded[0,:,ln_no],M_padded[1,:,ln_no], c=cm.rainbow((ln_no+0.01)/np.size(dB0)), lw=2, zorder=1)

    Mxy1, = ax1.plot (dB0,np.abs(M[0,:]+1j*M[1,:]),'k', lw=2, zorder=2)
    
    ax_scl1 = max(abs(np.linalg.norm(M,axis=0)))
    # set up the subplots as needed


    ax1.set_ylim(0, ax_scl1)
    ax1.set_xlim(dB0[0], dB0[-1])
    ax1.set_box_aspect(1)
    ax1.set_xticks([dB0[0],0,-dB0[0]]) 
    ax1.xaxis.set_ticklabels(['-1/TR','0','1/TR'])
    ax1.set_xlabel('Off-resonance')
    ax1.set_ylabel('$|M_{XY}|$')
    ax1.set_title('a) $|M_{XY}|$ across isochromats')
    #M_plot, = ax3.plot3D (M_padded[0,:],M_padded[1,:],M_padded[2,:],'r', lw=0.2)

        
    ax_scl2 = max(abs(np.abs(F)))*1.1
    ax2.set_ylim(0, ax_scl2)
    ax2.set_xlim(0.3*len(F), 0.7*len(F))
    ax2.set_box_aspect(1)
    ax2.set_xlabel('F-states')
    ax2.set_ylabel('$|FFT(M_{XY})|$')
    ax2.set_title('b) $|FFT(M_{XY})|$ across isochromats')
    
    return fig,ax1,ax2,tt,Mxy1

def update_Fstate_plots(dB0,M,ax1,ax2,Mxy1):
    
    F = np.fft.ifftshift(np.fft.fft(np.fft.fftshift(1j*M[0,:]+M[1,:])))/len(dB0)
    for dB0_idx in range(np.size(dB0)):
        ax_temp = ax1.get_lines()[dB0_idx]
        ax_temp.set_data(np.array([dB0[dB0_idx],dB0[dB0_idx]]),[0,np.abs(M[0,dB0_idx]+1j*M[1,dB0_idx])])
    Mxy1.set_data(dB0,np.abs(M[0,:]+1j*M[1,:]))

    ax2.clear()
    ax2.stem (np.arange(0,len(F)/2)-(len(F)/4),np.abs(F[1::2]),'k', markerfmt=" ",basefmt=" ")
    
    ax_scl = max(abs(np.abs(F)))*1.1
    ax2.set_ylim(0, ax_scl)
    ax2.set_xlim(-20, 20)
    ax2.set_box_aspect(1)
    ax2.set_xlabel('F-states')
    ax2.set_ylabel('$|FFT(M_{XY})|$')
    ax2.set_title('b) $|FFT(M_{XY})|$ across isochromats')
    
