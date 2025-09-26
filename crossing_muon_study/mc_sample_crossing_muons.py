import numpy as np
print("Imported NumPy")
import matplotlib
import matplotlib.pyplot as plt
print("Imported Matplotlib")
from matplotlib.colors import LogNorm
import matplotlib.patches as patches
print("Imported LogNorm")
import awkward as ak
import pickle 
import psutil
import os
import json
from matplotlib.patches import Circle, Patch
from matplotlib.collections import PatchCollection

import sys
sys.path.append("/afs/cern.ch/user/l/lcook")
from WCTE_event_display.EventDisplay import EventDisplay
from matplotlib import colors

print("Finished import")

# Load the data
data_path = "/eos/experiment/wcte/user_data/lcook/crossing_muon_study/wcsim_wCDS_mu+_Beam_672MeV_0cm_0001.npz"
data = np.load(data_path, allow_pickle=True)
print("Loaded data from", data_path)
print("Data keys:", data.files)
print("Hit dtype", data["digi_hit_charge"].dtype)
print("First hit array dtype", data["digi_hit_charge"][0].dtype)

#use truth information to identify crossing muon events
iev = 0
# data["track_start_position"][data["track_id"]==1]
regenerate_pickle_files = False
if regenerate_pickle_files == True:
    print("Convert numpy object arrays to awkward arrays")
    track_start_positions = ak.Array([list(x) for x in data["track_start_position"]])
    track_stop_positions = ak.Array([list(x) for x in data["track_stop_position"]])
    track_ids = ak.Array([list(x) for x in data["track_id"]])
    track_pids = ak.Array([list(x) for x in data["track_pid"]])
    
    digi_hit_charge = ak.Array([list(x) for x in data["digi_hit_charge"]])
    digi_hit_time = ak.Array([list(x) for x in data["digi_hit_time"]])
    digi_hit_pmt = ak.Array([list(x) for x in data["digi_hit_pmt"]])
    print("done")
    #save in pickle
    with open("track_info_ak_arrays.pkl", "wb") as f:
        pickle.dump({
            "track_start_positions": track_start_positions,
            "track_stop_positions": track_stop_positions,
            "track_ids": track_ids,
            "track_pids": track_pids,
            "digi_hit_charge":digi_hit_charge,
            "digi_hit_time":digi_hit_time,
            "digi_hit_pmt":digi_hit_pmt
        }, f)
    print("Saved track info awkward arrays to track_info_ak_arrays.pkl")
else:
    print("Loading pickled track info awkward arrays...")
    with open("track_info_ak_arrays.pkl", "rb") as f:
        track_data = pickle.load(f)
    track_start_positions = track_data["track_start_positions"]
    track_stop_positions = track_data["track_stop_positions"]
    track_ids = track_data["track_ids"]
    track_pids = track_data["track_pids"]
    digi_hit_charge = track_data["digi_hit_charge"]
    digi_hit_time = track_data["digi_hit_time"]
    digi_hit_pmt  = track_data["digi_hit_pmt"]
    print("Loaded track info awkward arrays from track_info_ak_arrays.pkl")

# print("PID of first track",ak.firsts(track_pids[track_ids==1]),len(ak.firsts(track_pids[track_ids==1])))
# print("PID of first track",ak.firsts(track_pids[track_ids==1]),len(ak.firsts(track_pids[track_ids==1])))
# print("Start of first track",ak.firsts(track_start_positions[track_ids==1]))

start_positions = ak.firsts(track_start_positions[track_ids==1])
stop_positions = ak.firsts(track_stop_positions[track_ids==1])
#draw straight line between start and stop positions
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X (cm)')
ax.set_ylabel('Y (cm)')
ax.set_zlabel('Z (cm)')
ax.set_title('Track ID 1 Trajectory')
for i in range(len(start_positions)):
    ax.plot([start_positions[i][0], stop_positions[i][0]], 
            [start_positions[i][1], stop_positions[i][1]], 
            [start_positions[i][2], stop_positions[i][2]], color='blue', label='Track ID 1')
# ax.scatter(start_positions[0][0], start_positions[0][1], start_positions[0][2], color='green', s=100, label='Start Position')
# ax.scatter(stop_positions[0][0], stop_positions[0][1], stop_positions[0][2], color='red', s=100, label='Stop Position')
# ax.legend()
#draw a cylinder of radius 100 cm around the y axis
u = np.linspace(0, 2 * np.pi, 100)
y = np.linspace(-271.4235, 271.4235, 100)
x = (307.5926/2) * np.outer(np.cos(u), np.ones_like(y))
z = (307.5926/2) * np.outer(np.sin(u), np.ones_like(y))
y = np.outer(np.ones_like(u), y)
ax.plot_surface(x, y, z, color='lightgrey', alpha=0.3)
ax.view_init(elev=13, azim=15, roll =90)  # Y-axis points up on screen

plt.savefig("track_id_1_trajectory.png")
plt.close()
print("Stop of track",ak.firsts(track_stop_positions[track_ids==1]))

#check to see whether tracks cross a plane 
#assume muon tagger is 70 cm after the end of the ID
z_pos_of_muon_tagger = (307.5926/2) + 70
x_pos_at_muon_tagger = start_positions[:,0] + (z_pos_of_muon_tagger - start_positions[:,2]) * (stop_positions[:,0] - start_positions[:,0]) / (stop_positions[:,2] - start_positions[:,2])
y_pos_at_muon_tagger = start_positions[:,1] + (z_pos_of_muon_tagger - start_positions[:,2]) * (stop_positions[:,1] - start_positions[:,1]) / (stop_positions[:,2] - start_positions[:,2])
y_pos_at_muon_tagger = y_pos_at_muon_tagger - start_positions[:,1]
print(start_positions[:,0])
print(x_pos_at_muon_tagger)
plt.scatter(x_pos_at_muon_tagger, y_pos_at_muon_tagger, alpha=0.5)
square = patches.Rectangle(
    (-25, -25),  # bottom-left corner
    50,               # width
    50,               # height
    edgecolor='blue',
    facecolor='none',          # transparent fill
    linewidth=2,
    label='Muon Tagger Area'  # label for the patch
)
# Add the square to the plot
ax = plt.gca()
ax.add_patch(square)
# Set equal aspect ratio
ax.set_aspect('equal', 'box')
plt.legend()
plt.xlabel("X position at muon tagger (cm)")
plt.ylabel("Y position at muon tagger (cm)")
plt.savefig("track_id_1_positions_at_muon_tagger.png")
plt.close()

# print("First 10 x",x_pos_at_muon_tagger[:10])
# print("First 10 y",y_pos_at_muon_tagger[:10])
# print("First 10 mask",)
muon_tagger_mask = (np.abs(x_pos_at_muon_tagger)<=25) & (np.abs(y_pos_at_muon_tagger)<=25)&(stop_positions[:,2]>(307.5926/2))

print("Number of crossing muon events", np.sum(muon_tagger_mask), "out of", len(muon_tagger_mask), "total events muon tagger cut",np.sum((np.abs(x_pos_at_muon_tagger)<=25) & (np.abs(y_pos_at_muon_tagger)<=25)))

#digi hit timing cuts
fig = plt.figure()
plt.hist(ak.flatten(digi_hit_time[muon_tagger_mask]),bins =np.linspace(0,50,100))
plt.xlabel("Digi hit times")
plt.ylabel("Hits")
plt.savefig("figs/mc_hit_timing.png")
plt.close()

# print(len(good_mpmt_channel_list),"wcte PMTs stably readout during the run",good_mpmt_channel_list)

hit_time_cut = (digi_hit_time>0) & (digi_hit_time<50)
total_charge=ak.sum(digi_hit_charge[hit_time_cut],axis=1)
fig = plt.figure()
plt.hist(total_charge[muon_tagger_mask],bins =50)
plt.xlabel("Total Charge")
plt.ylabel("Events")
plt.savefig("figs/mc_total_charge.png")
plt.close()

#get mapping from the event display function
eventDisplay = EventDisplay() 
eventDisplay.load_wcsim_tubeno_mapping("geofile_WCTE.txt")
unique_pmt_tube_nos = np.unique(ak.to_numpy(ak.flatten(digi_hit_pmt)))
print("min",min(unique_pmt_tube_nos),max(unique_pmt_tube_nos))
tube_numbers = np.arange(0,max(unique_pmt_tube_nos)+1)
#watchmal format tube numbers are -1 the wcsim tube number format so add on one
mPMT_id_map, pmt_pos_map = eventDisplay.map_wcsim_tubeno_to_slot_pmt_id(tube_numbers+1)
mpmt_slot_pmt_pos_lut = mPMT_id_map*100+pmt_pos_map
print("tube_numbers",len(tube_numbers),"last",tube_numbers[-1])
#map the digihit ids to wcte style
digi_hit_pmt_flat = ak.flatten(digi_hit_pmt)
digi_hit_pmt_flat_mapped = mpmt_slot_pmt_pos_lut[digi_hit_pmt_flat]
digi_hit_wcte_id = ak.unflatten(digi_hit_pmt_flat_mapped, ak.num(digi_hit_pmt))
#get good channel list from data
file_path = "/eos/experiment/wcte/data/2025_commissioning/processed_offline_data/production_v0_5/1610/run_1610_meta_data_json.json"
with open(file_path, "r") as f:
    metadata = json.load(f)
good_mpmt_channel_list = metadata["good_wcte_pmts"]
good_mpmt_channel_list = np.sort(good_mpmt_channel_list)

digi_hit_wcte_id_flat = ak.to_numpy(ak.flatten(digi_hit_wcte_id))
digi_hit_channel_mask_flat = np.isin(digi_hit_wcte_id_flat, good_mpmt_channel_list)
digi_hit_channel_mask = ak.unflatten(digi_hit_channel_mask_flat,ak.num(digi_hit_wcte_id))

if regenerate_pickle_files==True:
    #arrange the data by channel 
    mc_n_hits_array = np.zeros((len(digi_hit_pmt),len(good_mpmt_channel_list)), dtype=int)
    mc_charge_array = np.zeros((len(digi_hit_pmt),len(good_mpmt_channel_list)), dtype=int)
    mc_time_array = np.full((len(digi_hit_pmt),len(good_mpmt_channel_list)), np.nan, dtype=float)

    # fill the array: count hits per event per channel
    for i, channel in enumerate(good_mpmt_channel_list):
        if i%100==0:
            print("on channel", i, "of", len(good_mpmt_channel_list), "channel", channel)
        # create a mask for this channel
        channel_mask = (digi_hit_wcte_id == channel) & hit_time_cut
        # sum hits per event
        mc_n_hits_array[:, i] = ak.to_numpy(ak.sum(channel_mask, axis=1))
        mc_charge_array [:, i] = ak.to_numpy(ak.sum(digi_hit_charge[channel_mask], axis=1))
        times_for_channel = ak.fill_none(ak.min(digi_hit_time[channel_mask], axis=1), np.nan)
        mc_time_array[:, i] = ak.to_numpy(times_for_channel)
        
    with open("mc_data_np_arrays.pkl", "wb") as f:
        pickle.dump({
            "mc_n_hits_array": mc_n_hits_array,
            "mc_charge_array": mc_charge_array,
            "mc_time_array": mc_time_array,
            "good_mpmt_channel_list":good_mpmt_channel_list
        }, f)

else:
    with open("mc_data_np_arrays.pkl", "rb") as f:
        pickled_wcte_data = pickle.load(f)
        
    mc_n_hits_array = pickled_wcte_data["mc_n_hits_array"]
    mc_charge_array = pickled_wcte_data["mc_charge_array"]
    mc_time_array = pickled_wcte_data["mc_time_array"]
    good_mpmt_channel_list = pickled_wcte_data["good_mpmt_channel_list"]
    print("Pickled WCTE event data loaded.")
    
#check nhits>1
#create an instance of event diplay class
eventDisplay = EventDisplay() 
#load the positions of the mPMTs (an internal file to the event display class that 
#specificies where to plot and what orientation to plot mPMTs on the event display)
eventDisplay.load_mPMT_positions('mPMT_2D_projection_angles.csv')
pmt_slot = good_mpmt_channel_list//100
pmt_pos = good_mpmt_channel_list%100

data_to_plot = np.mean(mc_n_hits_array[muon_tagger_mask],axis=0)
ev_disp_data = eventDisplay.process_data(pmt_slot,pmt_pos,data_to_plot,sum_data=True)
eventDisplay.plotEventDisplay(ev_disp_data,color_norm=colors.Normalize(), style= "dark_background")
plt.savefig("figs/mc_mean_nhits_event_display.png",dpi =300)
plt.style.use('default') 
plt.close()

data_to_plot = np.mean(mc_charge_array[muon_tagger_mask],axis=0)
ev_disp_data = eventDisplay.process_data(pmt_slot,pmt_pos,data_to_plot,sum_data=True)
eventDisplay.plotEventDisplay(ev_disp_data,color_norm=colors.Normalize(vmin =0, vmax =15), style= "dark_background")
plt.savefig("figs/mc_mean_charge_event_display.png",dpi =300)
plt.style.use('default') 
plt.close()

## compare to data

print("Loading data pickled channel arrays...")
with open("crossing_muon_channel_arrays.pkl", "rb") as f:
    channel_data = pickle.load(f)
n_hits_array = channel_data["n_hits_array"]
charge_array = channel_data["charge_array"]
time_array = channel_data["time_array"]
unique_channels = channel_data["unique_channels"]
print("Loaded channel arrays from crossing_muon_channel_arrays.pkl")
print("are arrays equal",np.all(unique_channels==good_mpmt_channel_list))

#load mPMT_type
unique_mpmts = np.unique(unique_channels//100)
with open("/eos/experiment/wcte/wcte_tests/mPMT_led_events/dictionaries/other_mpmt_info.dict", "rb") as f:
    data = pickle.load(f)

tri_in_situ_list = []
tri_ex_situ_list = []
wut_in_situ_list = []
wut_ex_situ_list = []

for mpmt in unique_mpmts:
    mpmt_type = data[mpmt]['mpmt_type']
    mpmt_site = data[mpmt]['mpmt_site']
    
    if mpmt_type == 'Ex-situ' and mpmt_site == 'TRI':
        tri_ex_situ_list.append(mpmt)
    elif mpmt_type == 'In-situ' and mpmt_site == 'TRI':
        tri_in_situ_list.append(mpmt)
    elif mpmt_type == 'Ex-situ' and mpmt_site == 'WUT':
        wut_ex_situ_list.append(mpmt)
    elif mpmt_type == 'In-situ' and mpmt_site == 'WUT':
        wut_in_situ_list.append(mpmt)
    else:
        print("Bad mpmt site/type",mpmt_type,mpmt_site)
        raise Exception("Bad mpmt site/type",mpmt_type,mpmt_site)

eventDisplay = EventDisplay() 
eventDisplay.load_mPMT_positions('mPMT_2D_projection_angles.csv')
pmt_slot = good_mpmt_channel_list//100
pmt_pos = good_mpmt_channel_list%100
data_to_plot = (np.mean(charge_array, axis=0)/135.0)/np.mean(mc_charge_array[muon_tagger_mask],axis=0)
ev_disp_data = eventDisplay.process_data(pmt_slot,pmt_pos,data_to_plot,sum_data=True)
eventDisplay.plotEventDisplay(ev_disp_data,color_norm=colors.Normalize(vmin =0, vmax =2), style= "dark_background")
eventDisplay.label_mPMTs(np.arange(0,106))

#draw circles around mpmts
mpmt_groups = [tri_in_situ_list,tri_ex_situ_list,wut_in_situ_list,wut_ex_situ_list]
labels = ["Tri In-situ","Tri ex-situ","WUT In-situ","WUT ex-situ"]
linestyles = ["solid","dotted","solid","dotted"]
colors = ["white","white","green","green"]
scale = matplotlib.rcParams['figure.figsize'][0]/20
ax = plt.gca()
legend_handles=[]
for group, label, ls, color in zip(mpmt_groups, labels, linestyles, colors):
    # Extract the positions of this group
    positions = eventDisplay.mPMT_2D_projection[group, 1:3]

    # Make circle patches for this group
    circles = [Circle((pos[0], pos[1]), radius=0.48) for pos in positions]

    # Add them to the plot
    collection = PatchCollection(
        circles,
        facecolor="none",
        linewidths=2 * scale,
        edgecolors=color,
        linestyles=ls
    )
    ax.add_collection(collection)
    
    handle = Patch(
        facecolor="none",
        edgecolor=color,
        linestyle=ls,
        linewidth=2 * scale,
        label=label
    )
    legend_handles.append(handle)
    
ax.legend(handles=legend_handles)
# Add legend to identify groups
# ax.legend()
plt.savefig("figs/mc_data_charge_ratio.png",dpi=300)
plt.style.use('default') 
plt.close()

channel_by_channel_data_mc_ratio = (np.mean(charge_array, axis=0)/135.0)/np.mean(mc_charge_array[muon_tagger_mask],axis=0)

mpmt_groups = [tri_in_situ_list,wut_in_situ_list,wut_ex_situ_list,tri_ex_situ_list]
labels = ["Tri In-situ","WUT In-situ","WUT ex-situ","Tri ex-situ"]

for group, label in zip(mpmt_groups, labels):
    
    group_channel_mask = np.isin(unique_channels//100, np.array(group)) & ((unique_channels//100)!=37)
    print("There are",np.sum(group_channel_mask),"channels of type",label,"of total",len(group_channel_mask))
    
    plt.hist(channel_by_channel_data_mc_ratio[group_channel_mask], bins =np.linspace(0,1.7,50), label =label, alpha = 0.5)

plt.legend()
plt.title("Crossing Muon Sample")
plt.xlabel("Mean Charge Data to MC ratio")
plt.ylabel("Channels")
plt.savefig("figs/Channel_mc_data_ratio_excluding37.png")
plt.close()    
        
# #channel mean charge
# mean_charge_per_channel = np.mean(charge_array, axis=0)
# ev_disp_data = eventDisplay.process_data(pmt_slot,pmt_pos,mean_charge_per_channel,sum_data=True)
# eventDisplay.plotEventDisplay(ev_disp_data,color_norm=colors.LogNorm(), style= "dark_background")
# plt.savefig("temp_mean_charge_event_display.png")
# plt.style.use('default') 
# plt.close()

#load the angles file
angles_file = np.load("pmt_incident_angles.npz")
angle_pmt_chernenkov_mask = angles_file["angle_pmt_chernenkov_mask"]
angle_pmt_chernenkov = angles_file["angle_pmt_chernenkov"]
mpmt_id = angles_file["mpmt_id"]
pmt_id = angles_file["pmt_id"]
angle_glb_plt_id = mpmt_id*100+pmt_id

#check angle by pmt type

mpmt_groups = [tri_in_situ_list,wut_in_situ_list,wut_ex_situ_list,tri_ex_situ_list]
labels = ["Tri In-situ","WUT In-situ","WUT ex-situ","Tri ex-situ"]

for group, label in zip(mpmt_groups, labels):
    #mask over the angle list, pmt receives cherenlov light angle_pmt_chernenkov_mask, pmt is in the groups defined above, not 37 and is a well read out channel
    angle_channel_mask = np.isin(mpmt_id, np.array(group)) & (mpmt_id!=37) & np.isin(angle_glb_plt_id,good_mpmt_channel_list) & angle_pmt_chernenkov_mask
    print(label,"has",np.sum(angle_channel_mask),"channels in the ring")
    plt.hist(angle_pmt_chernenkov[angle_channel_mask], bins =50, label =label, alpha = 0.5)

plt.legend()
plt.title("Cherenkov Indicent Angle")
plt.xlabel("Cherenkov Indicent Angle [Cos theta]")
plt.ylabel("Channels")
plt.savefig("figs/IndicentCherenkovChannelType.png")
plt.close()     

mpmt_groups = [tri_in_situ_list,wut_in_situ_list,wut_ex_situ_list,tri_ex_situ_list]
labels = ["Tri In-situ","WUT In-situ","WUT ex-situ","Tri ex-situ"]

for group, label in zip(mpmt_groups, labels):
    #mask over the angle list, pmt receives cherenlov light angle_pmt_chernenkov_mask, pmt is in the groups defined above, not 37 and is a well read out channel
    group_channel_mask = np.isin(unique_channels//100, np.array(group)) & ((unique_channels//100)!=37)
    group_data_mc_ratio = channel_by_channel_data_mc_ratio[group_channel_mask]
    group_channel_ids = unique_channels[group_channel_mask]
    
    group_cos_angle = []
    group_angle_mask = [] #is inside chernekov ring
    for pmt_id in group_channel_ids:
        if np.sum(angle_glb_plt_id==pmt_id)!=1:
            print("problem with ids",np.sum(angle_glb_plt_id==pmt_id), pmt_id)
        group_cos_angle.append(angle_pmt_chernenkov[angle_glb_plt_id==pmt_id][0])
        group_angle_mask.append(angle_pmt_chernenkov_mask[angle_glb_plt_id==pmt_id][0])
    
    group_cos_angle = np.array(group_cos_angle)
    group_angle_mask = np.array(group_angle_mask)
            
    plt.scatter(group_cos_angle[group_angle_mask],group_data_mc_ratio[group_angle_mask],alpha = 0.7,label =label)
    # print("There are",np.sum(group_channel_mask),"channels of type",label,"of total",len(group_channel_mask))
    
    # plt.hist(channel_by_channel_data_mc_ratio[group_channel_mask], bins =np.linspace(0,1.7,50), label =label, alpha = 0.5)
    
    # angle_channel_mask = np.isin(mpmt_id, np.array(group)) & (mpmt_id!=37) & np.isin(angle_glb_plt_id,good_mpmt_channel_list) & angle_pmt_chernenkov_mask
    # print(label,"has",np.sum(angle_channel_mask),"channels in the ring")
    # plt.hist(angle_pmt_chernenkov[angle_channel_mask], bins =50, label =label, alpha = 0.5)

plt.legend()
# plt.title("Cherenkov Indicent Angle")
plt.grid(True,alpha=0.5)  # turn on gridlines
plt.xlabel("Cherenkov Incident Angle [Cos theta]")
plt.ylabel("Mean Charge Data to MC ratio")
plt.savefig("figs/ChargeAngleComparison.png", dpi =300)
plt.close()    