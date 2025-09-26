print("Start importing modules")
import pickle
import matplotlib.pyplot as plt
print("Imported matplotlib")
import numpy as np
print("Imported numpy")
import awkward as ak
print("Finished importing modules")

with open("crossing_muon_selection.pkl", "rb") as f:
    pickled_wcte_data = pickle.load(f)
    
selected_events = pickled_wcte_data["selected_events"]
print("Pickled WCTE event data loaded.",len(selected_events))
data = ak.Array(selected_events)

time_cut_low = 1680
time_cut_high = 1750
detector_hits = data["hit_mpmt_card_ids"]<125 #only WCTE channels
timing_mask = (data["hit_pmt_calibrated_times"]>time_cut_low) & (data["hit_pmt_calibrated_times"]<time_cut_high)
hit_pmt_readout_mask = data["hit_pmt_readout_mask"]==0
nhits = ak.num(data["hit_pmt_charges"][detector_hits & timing_mask & hit_pmt_readout_mask], axis=1)
charge = ak.sum(data["hit_pmt_charges"][detector_hits & timing_mask & hit_pmt_readout_mask], axis=1)

plt.figure()
plt.hist(np.array(charge), bins=np.linspace(4e5,6e5,50), label="Crossing Muon Charge")
plt.legend(loc="upper right")
plt.grid(True)   
plt.xlabel("WCTE Total Charge")  
plt.ylabel("Triggers")
# plt.yscale("log")
plt.title("WCTE Total Charge Distribution")
plt.savefig("figs/crossing_muon_charge_distribution.png")
plt.close()

wcte_hit_mask = detector_hits & timing_mask & hit_pmt_readout_mask
hit_channels = (100*data["hit_mpmt_slot_ids"]) + data["hit_pmt_position_ids"]
unique_channels = np.unique(np.array(ak.flatten(hit_channels[wcte_hit_mask])))
print("len unique channels hit", len(unique_channels))

load_pickle_data = True
if load_pickle_data==False:  
    print("Not loading pickled data, processing from scratch")
    n_hits_array = np.zeros((len(data),len(unique_channels)), dtype=int)
    charge_array = np.zeros((len(data),len(unique_channels)), dtype=int)
    time_array = np.full((len(data),len(unique_channels)), np.nan, dtype=float)

    # fill the array: count hits per event per channel
    for i, channel in enumerate(unique_channels):
        if i%100==0:
            print("on channel", i, "of", len(unique_channels), "channel", channel)
        # create a mask for this channel
        channel_mask = (hit_channels == channel) & wcte_hit_mask
        # sum hits per event
        n_hits_array[:, i] = ak.sum(channel_mask, axis=1)
        charge_array [:, i] = ak.sum(data["hit_pmt_charges"][channel_mask], axis=1) 
        # times_for_channel = data["hit_pmt_calibrated_times"][channel_mask]
        times_for_channel = ak.fill_none(ak.min(data["hit_pmt_calibrated_times"][channel_mask], axis=1), np.nan)
        time_array[:, i] = ak.to_numpy(times_for_channel)

    #save arrays as pkl
    with open("crossing_muon_channel_arrays.pkl", "wb") as f:
        pickle.dump({
            "n_hits_array": n_hits_array,
            "charge_array": charge_array,
            "time_array": time_array,
            "unique_channels": unique_channels
        }, f)
    print("Saved channel arrays to crossing_muon_channel_arrays.pkl")

if load_pickle_data:
    print("Loading pickled channel arrays...")
    with open("crossing_muon_channel_arrays.pkl", "rb") as f:
        channel_data = pickle.load(f)
    n_hits_array = channel_data["n_hits_array"]
    charge_array = channel_data["charge_array"]
    time_array = channel_data["time_array"]
    unique_channels = channel_data["unique_channels"]
    print("Loaded channel arrays from crossing_muon_channel_arrays.pkl")


#check for double hits
print(n_hits_array)
double_hit_mask = n_hits_array>1
n_double_hits = np.sum(double_hit_mask, axis=1)
print("Number of events with double hits:", np.sum(n_double_hits>0), "out of", len(n_hits_array))

#channel with highest total charge
total_charge_per_channel = np.sum(charge_array, axis=0)
print("len total_charge_per_channel", len(total_charge_per_channel))
highest_charge_channel_index = np.argmax(total_charge_per_channel)
highest_charge_channel = unique_channels[highest_charge_channel_index]
print("Channel with highest total charge:", highest_charge_channel, "with total charge", total_charge_per_channel[highest_charge_channel_index])

#histogram of channel 5501 charge distribution
channel_5501_index = np.where(unique_channels==5501)[0][0]
channel_5501_charge = charge_array[:, channel_5501_index]
plt.figure()
plt.hist(channel_5501_charge, bins = 50)
plt.xlabel("Charge on channel 5501")
plt.ylabel("Events")
plt.title("Channel 5501 Charge Distribution")
plt.savefig("temp_channel_5501_charge_distribution.png")
plt.close()

occupancy = np.mean(n_hits_array, axis=0)

import sys
sys.path.append("/afs/cern.ch/user/l/lcook")
from WCTE_event_display.EventDisplay import EventDisplay
from matplotlib import colors

#create an instance of event diplay class
eventDisplay = EventDisplay() 
#load the positions of the mPMTs (an internal file to the event display class that 
#specificies where to plot and what orientation to plot mPMTs on the event display)
eventDisplay.load_mPMT_positions('mPMT_2D_projection_angles.csv')
pmt_slot = unique_channels//100
pmt_pos = unique_channels%100
data_to_plot = occupancy
ev_disp_data = eventDisplay.process_data(pmt_slot,pmt_pos,data_to_plot,sum_data=True)
eventDisplay.plotEventDisplay(ev_disp_data,color_norm=colors.Normalize(), style= "dark_background")
plt.savefig("temp_occupancy_event_display.png")
plt.style.use('default') 
plt.close()

#channel mean charge
mean_charge_per_channel = np.mean(charge_array, axis=0)
ev_disp_data = eventDisplay.process_data(pmt_slot,pmt_pos,mean_charge_per_channel,sum_data=True)
eventDisplay.plotEventDisplay(ev_disp_data,color_norm=colors.LogNorm(), style= "dark_background")
plt.savefig("temp_mean_charge_event_display.png")
plt.style.use('default') 
plt.close()

mean_charge_per_channel = np.mean(charge_array, axis=0)
ev_disp_data = eventDisplay.process_data(pmt_slot,pmt_pos,mean_charge_per_channel,sum_data=True)
eventDisplay.plotEventDisplay(ev_disp_data,color_norm=colors.Normalize(vmin =0, vmax =2000), style= "dark_background")
plt.savefig("temp_mean_charge_event_display_norm.png")
plt.style.use('default') 
plt.close()