import uproot
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import awkward as ak
import pickle 
import psutil
import os


file_path    = "/eos/experiment/wcte/data/2025_commissioning/processed_offline_data/production_v0_5/1610/WCTE_offline_R1610S0_VME_matched.root"
tree_name    = "WCTEReadoutWindows"        # This will not change

# Open the file and grab the tree
f    = uproot.open(file_path)
tree = f[tree_name]


reference_ids = [31, 46]        # (TDC ref for IDs <31, ref1 for IDs >31)
t0_group     = [0, 1, 2, 3]       # must all be present
t1_group     = [4, 5, 6, 7]       # must all be present
t4_group     = [42, 43]           # must all be present
t4_qdc_cut   = 200                # Only hits above this value
act_eveto_group = [12, 13, 14, 15, 16, 17]   # ACT-eveto channels
act_eveto_cut = 1100
act_tagger_group = [18, 19, 20, 21, 22, 23]
hc_group = [9, 10]

hc_charge_cut = 150 # Only hits below this value
muon_tagger_group = [24, 25]
muon_tagger_cut = 125 # at least one of the two muon tagger channels must be above this value

# Load all four branches into NumPy arrays
branches = [
    "beamline_pmt_tdc_times",
    "beamline_pmt_tdc_ids",
    "beamline_pmt_qdc_charges",
    "beamline_pmt_qdc_ids",
]
regenerate_pickle_files = True #set false to regenerate pickled data
if regenerate_pickle_files==True : 
    print("Loading data...")
    data = tree.arrays(branches, library="ak")
    print(len(data["beamline_pmt_tdc_times"]), "events loaded")
        
    tdc_times = data["beamline_pmt_tdc_times"]
    tdc_ids   = data["beamline_pmt_tdc_ids"]

    # Correct TDC times based on reference channels
    mask_ref0 = tdc_ids == reference_ids[0]
    mask_ref1 = tdc_ids == reference_ids[1]
    ref0 = ak.firsts(tdc_times[mask_ref0])
    ref1 = ak.firsts(tdc_times[mask_ref1])
    corrected_tdc_times = ak.where(tdc_ids < reference_ids[0],tdc_times - ref0[:,None],tdc_times - ref1[:,None])
    #mask for triggers where reference is missing
    missing_reference_mask = (~ak.is_none(ref0)) & (~ak.is_none(ref1))

    #T0 Hits 
    print("Calculating T0 and T1 average times...")
    #make a numpy array of shape  (n_events, n_channels) with the earliest hit time for each channel in t0_group and t1_group
    t0_hit_times_per_channel = []
    for ch in t0_group:
        mask = tdc_ids == ch
        ch_times = corrected_tdc_times[mask]
        earliest = ak.firsts(ak.sort(ch_times))
        t0_hit_times_per_channel.append(earliest)
    # Convert list-of-arrays → array of shape (n_channels, n_events)
    t0_hits = ak.Array(t0_hit_times_per_channel)
    # Transpose to (n_events, n_channels)
    t0_hits = np.array(list(zip(*t0_hits.to_list())))
    #replace nones with nan
    t0_hits = np.where(t0_hits == None, np.nan, t0_hits).astype(float)
    have_t0_mask = np.all(t0_hits != np.nan, axis=1) 

    #T1 Hits 
    #make a numpy array of shape  (n_events, n_channels) with the earliest hit time for each channel t1_group
    t1_hit_times_per_channel = []
    for ch in t1_group:
        mask = tdc_ids == ch
        ch_times = corrected_tdc_times[mask]
        earliest = ak.firsts(ak.sort(ch_times))
        t1_hit_times_per_channel.append(earliest)
    # Convert list-of-arrays → array of shape (n_channels, n_events)
    t1_hits = ak.Array(t1_hit_times_per_channel)
    # Transpose to (n_events, n_channels)
    t1_hits = np.array(list(zip(*t1_hits.to_list())))
    #replace nones with nan
    t1_hits = np.where(t1_hits == None, np.nan, t1_hits).astype(float)
    have_t1_mask = np.all(t1_hits != np.nan, axis=1) 

    #calculate the mean where there are hits on all the channels, otherwise nan
    t0_avgs = np.mean(t0_hits, axis=1)
    t1_avgs = np.mean(t1_hits, axis=1)
    tof_vals = t1_avgs - t0_avgs

    event_mask = missing_reference_mask & have_t0_mask & have_t1_mask

    # ——— plot histograms ———
    plt.figure()
    plt.hist(t0_avgs[event_mask], bins=50)
    plt.xlabel("⟨T0⟩ (ns)")
    plt.ylabel("Counts")
    plt.title("T0 average time distribution")
    # plt.savefig("event_selection_crossing_muon_figures/T0AvTime.png")
    plt.close()

    plt.figure()
    plt.hist(t1_avgs[event_mask], bins=50)
    plt.xlabel("⟨T1⟩ (ns)")
    plt.ylabel("Counts")
    plt.title("T1 average time distribution")
    # plt.savefig("event_selection_crossing_muon_figures/T1AvTime.png")
    plt.close()

    plt.figure()
    plt.hist(tof_vals[event_mask], bins=1000)
    plt.xlabel("T1–T0 (ns)")
    plt.ylabel("Counts")
    plt.xlim(10,30)
    # plt.yscale("log")
    plt.title("TOF (T0 minus T1) distribution")
    # plt.savefig("event_selection_crossing_muon_figures/T0T1TimeDiff.png")
    plt.close()

    #ACT Eveto Hits
    print("Doing ACT-eveto...")
    qdc_charges = data["beamline_pmt_qdc_charges"] 
    qdc_ids     = data["beamline_pmt_qdc_ids"]
    act_eveto_hit_charges = []
    for ch in act_eveto_group:
        mask = qdc_ids == ch
        ch_charges = qdc_charges[mask]
        #take the first charge in the readout and add to list
        charge = ak.firsts(ch_charges)
        act_eveto_hit_charges.append(charge)
    # Convert list-of-arrays → array of shape (n_channels, n_events)
    act_eveto_hit_charges = ak.Array(act_eveto_hit_charges)
    act_eveto_hit_charges = np.array(list(zip(*act_eveto_hit_charges.to_list())))
    #replace nones with 0
    act_eveto_hit_charges = np.where(act_eveto_hit_charges == None, 0, act_eveto_hit_charges).astype(float)

    plt.figure()
    plt.hist(np.sum(act_eveto_hit_charges,axis=1), bins=100)
    #draw a vertical line at act_eveto_cut
    plt.axvline(x=act_eveto_cut, color='r', linestyle='--', label='Cut Value')
    plt.legend()
    plt.xlabel("ACT-eveto total QDC charge")
    plt.ylabel("Counts")
    plt.yscale("log")
    plt.title("Histogram of summed ACT-eveto charges")
    # plt.savefig("event_selection_crossing_muon_figures/ACTEvetoCharge.png")

    #ACT Tagger Hits
    print("Doing ACT tagger...")
    act_tagger_hit_charges = []
    for ch in act_tagger_group:
        mask = qdc_ids == ch
        ch_charges = qdc_charges[mask]
        #take the first charge in the readout and add to list
        charge = ak.firsts(ch_charges)
        act_tagger_hit_charges.append(charge)

    # Convert list-of-arrays → array of shape (n_channels, n_events)
    act_tagger_hit_charges = ak.Array(act_tagger_hit_charges)
    act_tagger_hit_charges = np.array(list(zip(*act_tagger_hit_charges.to_list())))
    #replace nones with 0
    act_tagger_hit_charges = np.where(act_tagger_hit_charges == None, 0, act_tagger_hit_charges).astype(float)
    plt.figure()
    plt.hist(np.sum(act_tagger_hit_charges,axis=1), bins=100)
    plt.xlabel("ACT-Tagger total QDC charge")
    plt.ylabel("Counts")
    plt.yscale("log")
    plt.title("Histogram of summed ACT-Tagger charges")
    # plt.savefig("event_selection_crossing_muon_figures/ACTTaggerCharge.png")

    #HC Hits
    print("Doing HC hits...")
    hc_hit_charges = []
    for ch in hc_group:
        mask = qdc_ids == ch
        ch_charges = qdc_charges[mask]
        #take the first charge in the readout and add to list
        charge = ak.firsts(ch_charges)
        hc_hit_charges.append(charge)

    # Convert list-of-arrays → array of shape (n_channels, n_events)
    hc_hit_charges = ak.Array(hc_hit_charges)
    hc_hit_charges = np.array(list(zip(*hc_hit_charges.to_list())))
    #replace nones with 0
    hc_hit_charges = np.where(hc_hit_charges == None, 0, hc_hit_charges).astype(float)
    plt.figure()
    #draw a vertical line at hc_charge_cut
    plt.axvline(x=hc_charge_cut, color='r', linestyle='--', label   ='Cut Value')
    plt.legend()
    plt.hist(np.sum(hc_hit_charges,axis=1), bins=100)
    plt.xlabel("HC total QDC charge")
    plt.ylabel("Counts")
    plt.yscale("log")
    plt.title("Histogram of summed HC charges")
    # plt.savefig("event_selection_crossing_muon_figures/HCCharge.png")

    #T4 Hits 
    print("Doing T4 hits...")
    #make a numpy array of shape  (n_events, n_channels) with the earliest hit time for each channel in t0_group and t1_group
    t4_hit_charges = []
    for ch in t4_group:
        mask = qdc_ids == ch
        ch_charges = qdc_charges[mask]
        #take the first charge in the readout and add to list
        charge = ak.firsts(ch_charges)
        t4_hit_charges.append(charge)
    # Convert list-of-arrays → array of shape (n_channels, n_events)
    t4_hit_charges = ak.Array(t4_hit_charges)
    # Transpose to (n_events, n_channels)
    t4_hit_charges = np.array(list(zip(*t4_hit_charges.to_list())))
    #replace nones with nan
    have_t4_mask = np.all(t4_hit_charges != None, axis=1) 
    t4_hit_charges = np.where(t4_hit_charges == None, 0, t4_hit_charges).astype(float)

    plt.figure()
    for i, ch in enumerate(t4_group):
        plt.hist(t4_hit_charges[:,i], bins=50, alpha=0.7, label=f"T4 Channel {ch}")
    # plt.hist(t4_hit_charges, bins=50)
    #draw a vertical line at T4_qdc_cut
    plt.axvline(x=t4_qdc_cut, color='r', linestyle='--', label   ='Cut Value')
    plt.legend()
    plt.xlabel("T4 Charges")
    plt.ylabel("Counts")
    plt.yscale("log")
    plt.title("Histogram of T4 charges")
    # plt.savefig("event_selection_crossing_muon_figures/T4Charge.png")

    #QDC readout failure
    print("Checking QDC readout...")
    plt.figure()
    n_qdc_hits = ak.num(qdc_ids)
    plt.hist(n_qdc_hits, bins=50)
    plt.xlabel("Number of QDC hits per event")
    plt.ylabel("Counts")
    plt.yscale("log")
    plt.title("Histogram of number of QDC hits per event")
    # plt.savefig("event_selection_crossing_muon_figures/NQDC.png")
    plt.close()
    #QDC mask where there are not 44 hits
    qdc_readout_mask = n_qdc_hits == 44

    #Muon Tagger
    print("Checking Muon Tagger...")
    muon_tagger_group = [24, 25]
    muon_tagger_hit_charges = []
    for ch in muon_tagger_group:
        mask = qdc_ids == ch
        ch_charges = qdc_charges[mask]
        #take the first charge in the readout and add to list
        charge = ak.firsts(ch_charges)
        muon_tagger_hit_charges.append(charge)
    # Convert list-of-arrays → array of shape (n_channels, n_events)
    muon_tagger_hit_charges = ak.Array(muon_tagger_hit_charges)
    muon_tagger_hit_charges = np.array(list(zip(*muon_tagger_hit_charges.to_list())))
    #replace nones with 0
    muon_tagger_hit_charges = np.where(muon_tagger_hit_charges == None, 0, muon_tagger_hit_charges).astype(float)
    plt.figure()
    plt.hist(muon_tagger_hit_charges[:,0], bins=np.linspace(0,2000,100), alpha=0.7, label="Left")
    plt.hist(muon_tagger_hit_charges[:,1], bins=np.linspace(0,2000,100), alpha=0.7, label="Right")
    plt.axvline(x=muon_tagger_cut, color='r', linestyle='--', label   ='Cut Value')
    plt.xlabel("Muon Tagger QDC charge")
    plt.legend()
    plt.ylabel("Counts")
    plt.yscale("log")
    plt.title("Histogram of Muon Tagger charges")
    # plt.savefig("event_selection_crossing_muon_figures/MuonTaggerCharge.png")
    plt.close()

    #pickle the masks and relevant arrays for later use
    with open("event_selection_data.pkl", "wb") as f:
        pickle.dump({
            "event_mask": event_mask,
            "qdc_readout_mask": qdc_readout_mask,
            "have_t4_mask": have_t4_mask,
            "tof_vals": tof_vals,
            "act_eveto_hit_charges": act_eveto_hit_charges,
            "hc_hit_charges": hc_hit_charges,
            "t4_hit_charges": t4_hit_charges,
            "act_tagger_hit_charges": act_tagger_hit_charges,
            "muon_tagger_hit_charges": muon_tagger_hit_charges,
        }, f)

#load pickled data
if regenerate_pickle_files==False:
    print("Loading pickled data...")
    with open("event_selection_data.pkl", "rb") as f:
        pickled_data = pickle.load(f)
        
    event_mask = pickled_data["event_mask"]
    qdc_readout_mask = pickled_data["qdc_readout_mask"] 
    have_t4_mask = pickled_data["have_t4_mask"]
    tof_vals = pickled_data["tof_vals"]
    act_eveto_hit_charges = pickled_data["act_eveto_hit_charges"]
    hc_hit_charges = pickled_data["hc_hit_charges"]
    t4_hit_charges = pickled_data["t4_hit_charges"]
    act_tagger_hit_charges = pickled_data["act_tagger_hit_charges"]
    muon_tagger_hit_charges = pickled_data["muon_tagger_hit_charges"]
    print("Pickled data loaded.")

#Do selection
print("Applying selection...")
data_quality_mask = event_mask & qdc_readout_mask & have_t4_mask
print(f"Data quality mask selects {np.sum(data_quality_mask)} / {len(data_quality_mask)} events")
eveto_mask = np.sum(act_eveto_hit_charges,axis=1)<act_eveto_cut
hc_mask = np.all(hc_hit_charges < hc_charge_cut, axis=1)
print("HC cut selects", np.sum(hc_mask & data_quality_mask), "/", np.sum(data_quality_mask), "events")
t4_cut = np.all(t4_hit_charges>t4_qdc_cut, axis=1)
print("T4 cut selects", np.sum(t4_cut & hc_mask & data_quality_mask), "/", np.sum(data_quality_mask), "events")

tof_mask = (tof_vals>10) & (tof_vals<17)
print("TOF mask selects", np.sum(tof_mask& t4_cut & hc_mask & data_quality_mask), "/", np.sum(data_quality_mask), "events")

muon_tagger = np.any(muon_tagger_hit_charges>muon_tagger_cut, axis=1)
print("Muon tagger cut selects", np.sum(muon_tagger & tof_mask & t4_cut & hc_mask & data_quality_mask), "/", np.sum(data_quality_mask), "events")  

act_group2_l = act_tagger_hit_charges[:,0]+act_tagger_hit_charges[:,2]+act_tagger_hit_charges[:,4]
act_group2_r = act_tagger_hit_charges[:,1]+act_tagger_hit_charges[:,3]+act_tagger_hit_charges[:,5]
cut_line_m = -1.0
cut_line_c = 2500

min_l_r_qdc = 600.0
group2_inv_cut = 1000.0         
act_group2_diagonal_cut = ((group2_inv_cut**2)/act_group2_l < act_group2_r) & (act_group2_l>min_l_r_qdc) & (act_group2_r>min_l_r_qdc)

#make some plots from selection
beam_selection = tof_mask& t4_cut & hc_mask & data_quality_mask
pass_eveto = eveto_mask & beam_selection
fail_eveto = (~eveto_mask) & beam_selection
pass_muon_selecton = muon_tagger & pass_eveto 
pass_muon_selecton_act_cut = act_group2_diagonal_cut & pass_muon_selecton

plt.figure()
act_group2_sum = np.sum(act_tagger_hit_charges,axis=1)
plt.hist(act_group2_sum[pass_eveto], bins=np.linspace(0,20000,1000), alpha=0.7, label="E_veto events")
plt.hist(act_group2_sum[pass_muon_selecton], bins=np.linspace(0,20000,1000), alpha=0.7, label="Muon tagger Selection")
plt.legend()
plt.xlabel("ACT Group 2 total QDC charge")
plt.ylabel("Counts")
plt.yscale("log")
plt.title("Histogram of summed ACT-Tagger charges for selection steps")
plt.savefig("event_selection_crossing_muon_figures/ACTTaggerCharge_Selection.png")

print("Plotting 2D histograms...")
cuts = [beam_selection,pass_eveto, pass_muon_selecton,pass_muon_selecton_act_cut]
cut_names = ["Beam_selection", "E_veto_events", "Muon_tagger","muon_selecton_act_cut"]
for i, (cut, cut_name) in enumerate(zip(cuts,cut_names)):
    print("Cut", cut_names[i], np.sum(cut))
    plt.figure()
    plt.hist2d(act_group2_l[cut], act_group2_r[cut], bins=(np.linspace(0,6000,300),np.linspace(0,6000,300)), norm=LogNorm(), cmap='viridis')
    if cut_name=="Muon_tagger" or cut_name=="E_veto_events":
        x = np.linspace(1,6000,100)
        y = (group2_inv_cut**2)/x
        y[y<min_l_r_qdc] = min_l_r_qdc
        x[x<min_l_r_qdc] = min_l_r_qdc
        plt.plot(x,y, color='r', linestyle='--', label='Diagonal Cut')
        plt.legend()    
    plt.xlabel("ACT Group 2 Left total QDC charge")
    plt.ylabel("ACT Group 2 Right total QDC charge")
    plt.title("ACT Group 2 Left vs Right QDC charge\n"+cut_name)
    plt.savefig("event_selection_crossing_muon_figures/group2_l_vs_r_"+cut_name+".png")
    plt.close()

plt.figure()
plt.hist(tof_vals[pass_eveto], bins=np.linspace(10,30,300), alpha=0.7, label="E_veto events")
plt.hist(tof_vals[pass_muon_selecton], bins=np.linspace(10,30,300), alpha=0.7, label="Muon tagger Selection")
plt.legend()
plt.xlabel("T0–T1 (ns)")
plt.ylabel("Counts")
plt.yscale("log")
plt.title("Histogram of TOF for selection steps")
plt.savefig("event_selection_crossing_muon_figures/TOF_Selection.png")
plt.close()
#### load WCTE data to apply cuts

time_cut_low = 1680
time_cut_high = 1750
branches = ["hit_mpmt_card_ids","hit_pmt_channel_ids","hit_mpmt_slot_ids","hit_pmt_position_ids","hit_pmt_charges","hit_pmt_calibrated_times","hit_pmt_readout_mask","window_data_quality"]
data = tree.arrays(branches, library="ak", entry_stop=10_000)
plt.figure()
plt.hist(ak.flatten(data["hit_pmt_calibrated_times"]),bins =np.linspace(1600,1800,400))
#draw vertical lines at time_cut_low and time_cut_high
plt.axvline(x=time_cut_low, color='r', linestyle='--', label='Low Cut')
plt.axvline(x=time_cut_high, color='g', linestyle='--', label='High Cut')
plt.legend()
plt.xlabel("WCTE Hit Times (ns)")
plt.ylabel("Counts")
plt.title("WCTE Hit Time Distribution")
plt.savefig("event_selection_crossing_muon_figures/WCTEHitTimes.png")
plt.close()

if regenerate_pickle_files==True:  
    start = 0
    event_total_charge = []
    event_nhits = []
    event_wcte_dq_mask = []

    for batch in tree.iterate(branches,step_size= 100_000, library="ak"):
        print("On batch starting at event", start)
        # mask_chunk = pass_muon_selection[start:start+len(batch)]
        detector_hits = batch["hit_mpmt_card_ids"]<125 #only WCTE channels
        timing_mask = (batch["hit_pmt_calibrated_times"]>time_cut_low) & (batch["hit_pmt_calibrated_times"]<time_cut_high)
        hit_pmt_readout_mask_batch = batch["hit_pmt_readout_mask"]==0
        nhits_batch = ak.num(batch["hit_pmt_charges"][detector_hits & timing_mask & hit_pmt_readout_mask_batch], axis=1)
        charge_batch = ak.sum(batch["hit_pmt_charges"][detector_hits & timing_mask & hit_pmt_readout_mask_batch], axis=1)
        
        event_nhits.extend(nhits_batch.to_list())
        event_total_charge.extend(charge_batch.to_list())
        
        event_wcte_dq_mask_batch= batch["window_data_quality"]==0
        event_wcte_dq_mask.extend(event_wcte_dq_mask_batch.to_list())
        
        start += len(batch)


    with open("wcte_event_data.pkl", "wb") as f:
        pickle.dump({
            "event_nhits": event_nhits,
            "event_total_charge": event_total_charge,
            "event_wcte_dq_mask": event_wcte_dq_mask,
        }, f)
        
if regenerate_pickle_files==False:
    print("Loading pickled WCTE event data...")
    with open("wcte_event_data.pkl", "rb") as f:
        pickled_wcte_data = pickle.load(f)
        
    event_nhits = pickled_wcte_data["event_nhits"]
    event_total_charge = pickled_wcte_data["event_total_charge"]
    event_wcte_dq_mask = pickled_wcte_data["event_wcte_dq_mask"]
    print("Pickled WCTE event data loaded.")
    

#plot the n hits and total charge separately for events that pass and fail the muon selection
vetoed_electron_selection = fail_eveto & event_wcte_dq_mask & beam_selection
electron_selection = (np.sum(act_eveto_hit_charges,axis=1)>2000) & event_wcte_dq_mask & beam_selection
muon_selection = act_group2_diagonal_cut & event_wcte_dq_mask & beam_selection & pass_eveto 
pion_selection = (~act_group2_diagonal_cut) & event_wcte_dq_mask & beam_selection & pass_eveto
muon_tagger_selection = act_group2_diagonal_cut & event_wcte_dq_mask & beam_selection & pass_eveto & muon_tagger
print("muon selection selects", np.sum(muon_tagger_selection), "/", np.sum(event_wcte_dq_mask & beam_selection), "events")

plt.figure()
plt.hist(np.array(event_nhits)[event_wcte_dq_mask & beam_selection], bins=np.linspace(0,1400,100) , histtype="step", label="Beamline cuts and DQ cuts")
plt.hist(np.array(event_nhits)[vetoed_electron_selection], bins=np.linspace(0,1400,100), histtype="step", label="Vetoed Electron Selection")
plt.hist(np.array(event_nhits)[electron_selection], bins=np.linspace(0,1400,100), alpha=0.7, label="Electron Selection (higher ACT group 1 cut)")
plt.hist(np.array(event_nhits)[muon_selection], bins=np.linspace(0,1400,100), alpha=0.7, label=""+r"$\mathrm{ACT_{group2}}$"+" Muon Selection and TOF Cut")
plt.hist(np.array(event_nhits)[pion_selection], bins=np.linspace(0,1400,100), histtype="step", label=""+r"$\mathrm{ACT_{group2}}$"+" Pion Selection and TOF Cut")
plt.hist(np.array(event_nhits)[muon_tagger_selection], bins=np.linspace(0,1400,100), alpha=0.7, label="Muon Tagger and "+r"$\mathrm{ACT_{group2}}$"+" Muon Cut and TOF Cut")
plt.xlim((0,2000))
plt.ylim((0,20000))
plt.legend(loc="upper right")
plt.grid(True)   
plt.xlabel("WCTE Number of Hits")
plt.ylabel("Triggers")
# plt.yscale("log")
plt.title("WCTE Number of Hits Distribution")
plt.savefig("event_selection_crossing_muon_figures/WCTENHits_Selection.png")
plt.close()

plt.figure()
plt.hist(np.array(event_total_charge)[event_wcte_dq_mask & beam_selection], bins=np.linspace(0,1e6,100), histtype="step", label="Beamline cuts and DQ cuts")
plt.hist(np.array(event_total_charge)[vetoed_electron_selection], bins=np.linspace(0,1e6,100), histtype="step", label="Vetoed Electron Selection")
plt.hist(np.array(event_total_charge)[electron_selection], bins=np.linspace(0,1e6,100), alpha=0.7, label="Electron Selection (higher ACT group 1 cut)")
plt.hist(np.array(event_total_charge)[muon_selection], bins=np.linspace(0,1e6,100), alpha=0.7, label=""+r"$\mathrm{ACT_{group2}}$"+" Muon Selection")
plt.hist(np.array(event_total_charge)[pion_selection], bins=np.linspace(0,1e6,100), histtype="step", label=""+r"$\mathrm{ACT_{group2}}$"+" Pion Selection")
plt.hist(np.array(event_total_charge)[muon_tagger_selection], bins=np.linspace(0,1e6,100), alpha=0.7, label="Muon Tagger and "+r"$\mathrm{ACT_{group2}}$"+" Muon Cut")
plt.xlim((0,1.5e6))
plt.ylim((0,15000))
plt.legend(loc="upper right")
plt.grid(True)   
plt.xlabel("WCTE Total Charge")  
plt.ylabel("Triggers")
# plt.yscale("log")
plt.title("WCTE Total Charge Distribution")
plt.savefig("event_selection_crossing_muon_figures/WCTETotalCharge_Selection.png")

#investiagate electrons with muon-like wcte charge
electron_wcte_muon_like_mask = electron_selection & (np.array(event_total_charge)>4.5e5) & (np.array(event_total_charge)<5.5e5)
plt.figure()
plt.hist(np.sum(act_eveto_hit_charges,axis=1)[electron_selection], bins=np.linspace(0,2500,100), alpha=0.7, label="E_veto Events")
plt.hist(np.sum(act_eveto_hit_charges,axis=1)[electron_wcte_muon_like_mask], bins=np.linspace(0,2500,100), alpha=0.7, label="E_veto events with muon-like WCTE charge")
plt.yscale("log")
plt.legend()
plt.xlabel("ACT Group 1 total QDC charge")
plt.ylabel("Triggers")
plt.title("ACT Group 1 total QDC charge for E_veto events with muon-like WCTE charge")
plt.savefig("event_selection_crossing_muon_figures/ACTEvetoCharge_ElectronMuonLike.png")
plt.close()

print(f"Memory usage: {process.memory_info().rss / 1024**2:.2f} MB")

if regenerate_pickle_files==True:  
    start = 0
    
    selected_events = []
    branches = ["hit_mpmt_card_ids","hit_pmt_channel_ids","hit_mpmt_slot_ids","hit_pmt_position_ids","hit_pmt_charges","hit_pmt_calibrated_times","hit_pmt_readout_mask","window_data_quality","readout_number"]

    for batch in tree.iterate(branches,step_size= 100_000, library="ak"):
        #get the muon tagger selection mask for this batch
        batch_indices = np.arange(start, start+len(batch))
        muon_tagger_selection_batch = muon_tagger_selection[batch_indices]
        selected_batch = batch[muon_tagger_selection_batch]
        selected_events.extend(selected_batch.to_list())
        start += len(batch)
        print("On event", start, "selected", len(selected_events), "so far")
        print(f"Memory usage batch loip: {process.memory_info().rss / 1024**2:.2f} MB")
    # selected_events = ak.concatenate(selected_events, axis=0)
    print(len(selected_events), "events selected")
    # Save to pickle
    with open("crossing_muon_selection.pkl", "wb") as f:
        pickle.dump({"selected_events": selected_events}, f)

    print(f"Saved {len(selected_events)} events to selected_events.pkl")
    
        
if regenerate_pickle_files==False:
    print("Loading pickled WCTE event data...")
    with open("crossing_muon_selection.pkl", "rb") as f:
        pickled_wcte_data = pickle.load(f)
        
    selected_events = pickled_wcte_data["selected_events"]
    print("Pickled WCTE event data loaded.")