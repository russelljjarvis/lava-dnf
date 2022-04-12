#!/usr/bin/env python
# coding: utf-8

# 
# <h1 align="center">
# Detailed Update + Opinion Seeking Plan/Presentation
# </h1>
# 
# * Not a true to form detailed update 
# * A kind of planning and brain storming presentation
# * I would really benefit from any input (now or delayed), this is online and contact me on teams.
# 

# In[1]:


import potjans_as_network
import pandas as pd
from tutorial06_hierarchical_processes import plot_conn_sankey, generate_sankey_figure


# <h3> started position 2nd-3rd of February. 
# today is12 April (a bit more than 2 months) </h3> 
# 
# <h4> Position Description (apparently not a super serious contract) </h4>
# 
# 
# - [x] Teaching, Developing course work
# - [ ] Develop a neuromorphic interface using Julia+Python **(more on this)**
# - [ ] Develop novel event based learning algorithms for acting on spatio-temporal sensory data **(more on this)**
# - [ ] Co-supervision honors/post-grad **(admitted to HDR recently)**
# - [ ] Implement a model of cortical neural processing on neuromorphic hardware using the above interface **(more on this)**
# - [ ] Develop novel event based learning algorithms for acting on spatio-temporal sensory data **I have been looking into improving Odessa.jl**
# - [ ] Researching and Collaboration
# - [ ] Grant Writing
#     
#     
#     
# 
# <!--Develop a neuromorphic interface using Julia+Python [possible]
# - found appropriate connectomes
# - Done exploratory work in Lava (a new Intel supported Loihi interface)
# Develop novel event based learning algorithms for acting on spatio-temporal sensory data.
# - Trying to improve Odesa.jl 
# Use the interface to solve visual tracking and tempo spatial sequence classification problem.
# - No
# Implement a model of cortical neural processing on neuromorphic hardware using the above interface.
# - No
# Teaching, Developing course work [x]
# Researching and Collaboration. []
# Grant Writing. [No]
# Co-supervision honors/post-grad (joined HDR) []
# 
# 
# 
# 
# | <h1> Position Description  </h1> | 
# | ------------- |
# | - [ ] Develop a neuromorphic interface using Julia+Python  |
# | col 2 is      |
# | zebra stripes |
# | <ul><li>item1</li><li>item2</li></ul>| 
# # Position Description (apparently not a super serious contract).
# 
# -->
#     
# 

# <h1>  Teaching. </h1> 
# 
# I made a lecture/tutorial for MoNE.
# 
# ![]()
# 
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BBKqg6ng23DWC4z-S5o_oiekM-vFHvgp?usp=sharing)
# 
# <h2> Why Google Colabatory? </h2>
# 
# * Cloud Environment
# * Highly reproducible (If code runs for you it runs for everyone). 
# * It is a multi language platform that can also render HTML and $ \LaTeX $ 
# * Potential to build greater continuity between Neurons in Action (NIA) course and my tutorial
# 
# Future Absorb some of NIA into google colab 
# 
# 
# 

# <h1 align="center">
# 
#     Julia Odesa
# </h1>
# 
# <h2 align="center">
#  Why Julia 
# </h2>
# 
# * Speed and resource friendliness.
# * It was my goal to show that the julia version of convolutional Odessa could be made faster.
# * Intended to show I could speed up Python code.
# * I refactored the code to make it follow Julia+Github conventions.
# * Broke code into smaller functions, added readme documentation.
# 
# <!--# Profile view. recover from chat with Yesh.-->
# 
# 
# 
# <h1 align="center">
# <img src="30minupdate/odesa_jl.png" 
#      width="800" 
#      height="650" />
# </h1>
# 

# 
# I found that a record function had a triply nested for loop and it was the only part of the code that was not being optimized.
# 
# <h1 align="center">
# <img src="30minupdate/profileview2.png" 
#      width="450" 
#      height="100"/>
# </h1>
# 

# 
#  
# ```julia 
#     function iter_record!(out_x, out_y,ts,layer::Conv,)
#         lx,ly,rx,ry = setup_record(layer::Conv, out_x, out_y, ts)
#         for y ∈ ly:ry, x ∈ lx:rx, n ∈ 1:layer.nNeurons
#             record!(Int32(n), Int32(x), Int32(y),ts,layer)
#         end
#     end
# ```
# 

# 
# 
# ```julia
# 
#     function record!(n::Int32, x::Int32, y::Int32,ts::Int32,layer::Conv)
#         scalar_to_compare::Float32 = exp((layer.winnerTrace[n, x, y] - ts) / layer.traceτ)
#         if scalar_to_compare >= 0.1
#             layer_w_n = view(layer.w,:, n)
#             layer_Δ_n = view(layer.Δ,:, n, x, y)
#             layer_w_n .= (1.0 - layer.η) .* layer_w_n .+ layer.η .* layer_Δ_n
#             layer_w_n .= layer_w_n ./ norm(layer_w_n)
#             layer.thresh[n] =
#                 (1.0 - layer.threshη) * layer.thresh[n] + ...
#                 layer.threshη * layer.ΔThresh[n, x, y]
#         else
#             if exp((layer.noWinnerTrace[x, y] - ts) / layer.traceτ) >= 0.1
#                 layer.thresh[n] = layer.thresh[n] - layer.thresholdOpen
#             end
#         end
#     end
#  ```

# 
# <!--<h1 align="center">
# <img src="30minupdate/profile_view.png" 
#      width="400" 
#      height="250" />
# </h1>-->
# 
# 
# Yesh showed that on a simple fully connected model trained on Iris classification. 
# <h3> Yesh found that Julia trained on Iris flower classification </h3>
# 
# * 38 Python/1.45 Julia
# * Julia 26 * speed up.
# * Classified Correct Julia : 0.92, Python :0.86
# 
# 
# Both the Python version and the Julia version of convolution are slow
# 
# * Applied profile view
# 
# * Found that a triply nested for loop in the record function is one of the slowest parts of the algorithm.
# 
# * reduced precision from Float64 to Float32.
# 
# * experimented with ways to speed up algorithm. Did not succeed yet, but I have some good leads.
# 
# 

#  
# <h3> Julia Speed Leads: </h3>
# 
# * Loop Vectorization (a module)
# * Linear Indexing
# * Allocate Mutable Static sized Arrays inside small functions
# * Multithreading
# 

# # Finally I have ran some stuff
# * **I ran a Cpp+CUDA model on NVIDIA jetson nano**
# * Since then we have decided to focus on Lava.
# 
# * I ran the multi area model on a my secondhand ebay gaming workstation (memory exhaustion).
# 

# <!--
# <h1 align="center">
# <img src="30minupdate/visual_schem.png" 
#      width="350" 
#      height="400" />
# </h1>
# -->
# <h2 align="center">
#   For those of you who don't know, so what is my project anyway? 
# </h2> 
# 
# <h3 align="center">
# 
# Introduction to My Main Project: An FPGA Bio-net model
# </h3>
# 
# 
# Problem in neuroscience
# We still don’t understand the contributions to learning of the brains large scale.
# Human V1 140million in each hemisphere.
# 280,000,000 neurons (both hemispheres) 
# 
# Human Brain 100Billion (electrical) neurons (100,000,000,000) neurons
# 
# <h1 align="center">
# <img src="30minupdate/cropped_vis_schem_simple2.png" 
#      width="450" 
#      height="500" />
# </h1>
# 
# 
# <!--
# # Usefulness of Lava for large scale biological modelling work
# 
# #### What we want from an interface
# 
# - [x] Ability to define LIF Cell populations.
# - [x] Means to specify forwards connectivity between layers as populations.
# - [x] Means to specify recurrent connectivity whithin a population.
# - [x] Inhibitory Synapses (negative weight values possible)
# - [x] Capacity to support high cell counts.
# 
# #### Not yet in the interface
# - [ ] Spike Timing Dependent Plasticity (STDP), or on chip local learning rules (coming).  
# - [ ] An ability to specify synaptic delays (hyper-synchronous epileptic network activity).  
# - [ ] Ability to visualize the layered architecture (nothing like TorchViz or TensorBoard for ANN architecture yet).
# - [ ] Delay Learning (possibly not a planned addition)
# - [ ] Adaptive Neurons (supported by SLAYER hard to make interoperable)
# - [ ] performance profiling (including power consumption). (coming)
# -->

# 
# 
# <!--
# <h1 align="center">
# <img src="30minupdate/lateral_inhibition.png" 
#      width="350" 
#      height="400" />
# </h1>
# -->
# <h1 align="center"> 
# Mark has created Neural Engines which is the FPGA progression of work on 
# the Orca DSP chip.
# </h1>
# 
# * Ideally the Neural Engine would be a bit like Loihi gen 3.
# * Its strenghts are it supports a very high neuron and synapse count. 
# 
# <h3 align="center"> FPGA Implementation </h3>
# <!--
# <h1 align="center">
# <img src="30minupdate/one_to_one_connectivity.png" 
#      width="350" 
#      height="400" />
# </h1>
# -->
# <h3> Why is this non-trivial?</h3>
# * Its limitations are its synaptic weight bit precision, and its $ V_{M} $ bit precision
# 
# 
# <h1 align="center">
# <img src="30minupdate/low_bit_precision1.png" 
#      width="300" 
#      height="450" />
# </h1>
# 
# * Synaptic weights can obtain very high values.
# 
# 

# 
# <h1 align="center">
# <img src="30minupdate/low_bit_precision2.png" 
#      width="300" 
#      height="450" />
# </h1>
# 

# 
# <h1> Biological network models can be expressed as a graph: a function of Vertices and Edges </h1>
# 
# <h1 align="center">
#  $G(V, E)$
# </h1>
# 
# Below is a diagram of the Potjan's cortical model. This model can be thought of as the composition of many weighted directed graphs, therefore we will use Lava a supported interface to begin to build a cortical model with the Python Loihi simulator.
# 
# <h1 align="center">
# 
# <img src="Schematic-diagram-of-the-Potjans-Diesmann-cortical-microcircuit-model.png" 
#      width="300" 
#      height="350" />
# 
# </h1>
# 
# 

# 
#  
#  
# <h1 align="center">
# <img src="30minupdate/multi_area_connectome.png" 
#      width="500" 
#      height="750"/>
# </h1>
# <h1 align="center">
# 
# * Each node below is a mini actually a Potjan network.
# </h1>
# 
# <h1 align="center">
# <img src="30minupdate/multi_area.png" 
#      width="500" 
#      height="750" />
# </h1>
# 
# 

# # Two parts to the project
# <h3> Part 1 </h3> Build an interface to Neural Engines on FPGA (Orca version 2). (Engineering year 1)
# First year is just establishing the interface
# <h3> Part 2 </h3> Doing hypothesis testing on the resulting model (Orca version 2). (Science year 2)
# First year is just establishing the interface
# 

# <h1>
#     Population Diagram
# </h1>
# 
# <h4>
#     This Sankey Diagram Network diagram is potentially programable with sliders that control population size (nodes) and connection probability ribon width
# </h4>
# 
# * **weight values are $ \times $ 100 for visibility
# 
# * **node sizes are $ \times $ 100 for visibility
# 

# In[2]:


G = potjans_as_network.G

list_of_dicts=[]
cnt=0
for edge in G.edges:
    
    list_of_dicts.append({'src':edge[0],'tgt':edge[1],'weight':potjans_as_network.weights[cnt]})
    cnt+=1
df = pd.DataFrame(list_of_dicts)
fig = generate_sankey_figure(list(G.nodes),df)


# In[3]:


import markdown
from IPython.display import display, Markdown, Latex, HTML


# In[4]:


#display(Markdown('# Potjans network topology'))
#fig.show()


# Proposed Plan
# * Build very small FPGA neural Engine with only $10,000~50,000$ neurons on FPGA.
# * 2 columns.
# * Small number needed for fast loading network on, and fast reading spikes off.
# * program synaptic densities with sliders.
# * ie E/I ratio could be a slider.

# In[5]:


# Set default parameters 
e_i_ratio = 0.0

import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, HBox, VBox, Layout

button_layout = Layout(width='180px', height='30px')

slider_total_width = '800px'
slider_desc_width = '200px'
ei_slider = widgets.FloatSlider(value = e_i_ratio, min = 0., max = 0.5, step = 0.15, 
                                 description = 'Charging EI ratio:', 
                                 readout_format = '.1f', 
                                 continuous_update = False, 
                                 layout = {'width': slider_total_width}, 
                                 style = {'description_width': slider_desc_width})

def change_connections(e_i_ratio=0.0):
    return e_i_ratio

ei_slider = widgets.FloatSlider(value = e_i_ratio, min = 0., max = 0.5, step = 0.15, 
                                 description = 'Charging EI ratio:', 
                                 readout_format = '.1f', 
                                 continuous_update = False, 
                                 layout = {'width': slider_total_width}, 
                                 style = {'description_width': slider_desc_width})

def change_l4(e_i_ratio=0.0):
    return e_i_ratio


# In[6]:



nt = potjans_as_network.interactive_population(potjans_as_network.node_name, potjans_as_network.G,potjans_as_network.weights,potjans_as_network.cd)


# In[7]:


display(Markdown('# Potjans network topology'))

main_widgets = interactive(change_connections, tau = ei_slider)
display(VBox(children=[main_widgets]))
fig.show()


# 
# <h1 align="center">
# <img src="30minupdate/raster_plot_out.png"
#      width="650" 
#      height="700" />
# </h1>
# 

# In[8]:


display(Markdown('# Potjans network topology'))

nt.show('population.html')


# ### Building the Inteface
# * Instantiate Potjans model on Loihi using Lava.
# * Problems, Lava itself is incomplete, and doesn't support STDP or synaptic delays yet.
# * Interface should be compatible with Intels existing tools.
# * Lava 2 Loihi compiler is **closed source**

# <h2> I have explored which connectomes should I use for the Loihi+Neural Engines simulation? </h2>
# 
# - [x] PyNNs OpenSource Brain implementation of Potjans and Diesmand.
# - [x] GENNs Multi area model (contains above models).
# - [ ] Allen Brain V1 BMTK, Sonata (hard to track down a non-distance+cell morphology based wiring scheme).
# - [ ] Existing Orca Potjan's model expressed in Teili/Brian2 (complicated build/code dependency hell)
# 
# 
# Each connectome wil be modified by increasing the number of pre-synaptic and post-synaptic connections and then by for pruning and simulated neuronal development.
# 

# <!--
# ![30minupdate/multi_area_connectome.png](30minupdate/springs.jpeg)
# ![30minupdate/multi_area_connectome.png](30minupdate/LoihiImage.png)
# -->
# 
# 
# <h1> Agile Frame Combined with Open Source Tools </h1>
# 
# * Start small entry level FPGA (?) buildable with FOSS tools.
# * First establish (proof of concept) very basic functionality.
# * Schedule in a small number of additional features periodically.
# * Scale up incrementally?.
# 
# 
# ## Versus jump to largest scale system ?
# 
# <!--<h1 align="center">
# <img src="30minupdate/springs.jpeg" 
#      width="300" 
#      height="450" />
# </h1>-->
# 
# 
# <h1 align="center">
# <img src="30minupdate/LoihiImage.png" 
#      width="450" 
#      height="600" />
# </h1>
# 

# ## Problem Open source software in HDL context is rare.
# 
# * A Collective Action Problem.
# 
# * Intel Loihi compiler is closed source.
# 
# * HDL code is typically closed source
# 
# In some ways we would like to create Loihi 3.0x.
# 
# * Putting HDL code on GitHub might signal williness to cooperate

# 
# <h3>    Reduced bit precision </h3>
# Its a hardware design issue, but its also related to ANN Network Compression 
# * means reducing on bit precision of synaptic weights, so the weights are quantized (they leap between values).
# <!--* Reducing the time and resources needed to converge an ANN model.
# * Building a large SNN modelin hardware seems to need the same constraints.-->
# 

# 
# 
# <img src="30minupdate/attractor1.png" 
#      width="500" 
#      height="650" />

# 
# * In Pablo's simulations reducing bit precision has had bad effects on the even distribution of synaptic weights.
# * Having weight values that are dispersed is important for encoding a wide number of attractor states.
# 
# 

# 
# <!--
# <h1 align="center">
# <img src="30minupdate/attractor2.jpeg" 
#      width="500" 
#      height="650" />
# </h1>
# -->

# 
# <h3 align="center">
#     Brunel's predictions about synaptic weight distributions
# </h3>
# <h4 align="center">
#  synaptic connectivity matrix has the following properties:
# </h4>
# 
# * Connectome is sparse, with a large fraction of zero synaptic weights ('potential' synapses);
# * Bidirectional reciporicated connections.
# * Bidirectionally coupled pairs of neurons are over-represented in comparison to a random network;
# * Bidirectionally connected pairs have stronger synapses on average than unidirectionally connected pairs. 
# * These features reproduce quantitatively available data on connectivity in cortex. 
# * ... and it suggests connectivity in cortex is optimized to store a large number of attractor states in a robust fashion.    
# 
# 

# 
# <h3> Building a very large scale FPGA brain simulator would require  reduced bit precision </h3>
# Its a hardware design issue, but its also related to ANN Network Compression 
# * means reducing on bit precision of synaptic weights, so the weights are quantized (they leap between values).
# <!--* Reducing the time and resources needed to converge an ANN model.
# * Building a large SNN modelin hardware seems to need the same constraints.-->
# 
# <h3> EEG Left self-sustained asynchronous-irregular activity middle epilepsy activity </h3>
# 
#     
# <h1 align="center">
#     <img src="30minupdate/epilepsy.jpg" 
#      width="500" 
#      height="650" />
# </h1>
#     
#     

# <h3> Left self-sustained asynchronous-irregular activity right epilepsy activity </h3>
# 
# <h1 align="center">
#     <img src="30minupdate/epilepsy.gif" 
#      width="400" 
#      height="550" />
# </h1>
# 
# 
# 
# 

# In[ ]:





# In[9]:


#main_widgets = interactive(change_connections, tau = ei_slider)
#display(VBox(children=[main_widgets]))


# 
# 
# Protocol 1
# 
# <h1 align="center">
# <img src="30minupdate/regular_stdp.jpeg" 
#      width="300" 
#      height="450" />
# </h1>
# 
# 

# 
# ## Prevent Epilepsy
# * Optimize the STDP shape?
# * Clip Potentiation maximum value?
# *Exagerrated Depression Phase
# 
# Protocol 2
# 
# <h1 align="center">
# <img src="30minupdate/enhanced_depression_phase_clipped.jpeg" 
#      width="300" 
#      height="450" />
# </h1>
# 
# Can we optimize the length of applying protocol 2?
# 
# * Can I use parameter optimization on synaptic depression strength to make a limited bit precision network  behave like a reputable CPU biologically plausible model?
# 

# <!--Doing science is hard and its possible I will spend a lot of time getting the interface to work.-->
# 
# <h1> A Brainstorm of Scientific Ideas to Test </h1>
# 
# * Probably not enough time in 2 years, but helpful for grant writing, and getting Loihi.
# 
# * Weight multiplication and clipping could mean unrealistic network behavior. The only viable exercise might be optimizing/exploring which parameters lead to realistic/useable weight evolution.
# 
# * If a large scale simulation is viable due to realistic balanced weight growth:
# 
# * Plot the Trajectory of neuronal development under different initializations.
# 
# * Can different initializations of STDP based pruning reliably lead  back to same behavior?
# 
# * Starting from a hyper connected network is there an optimal time to stop a simulation of pruning.
# 
# * Dynamics (firing frequencies).
# 
# * Is predictive coding possible?
# 
# <!--
# * a limited bit precision network can do predictive coding pattern completion at large scale (corroborate Pablo's thesis)
# 
# 
# * Measure number of discernable attractors in converged network simulations. 
# -->

# <h1 align="center">
# <img src="30minupdate/pruning4.jpeg" 
#      width="400" 
#      height="550" />
# </h1>
# 
# 
# <h1 align="center">
# <img src="30minupdate/pruning1.jpeg" 
#      width="500" 
#      height="650" />
# </h1>
# 
# 
# 
# 

# <!--
# ## We shouldn't be surprised if the very large scale network behavior is not useful for creating attractors.
# 
# ### If I had to guess I would say that self regulation will not emerge via being large scale alone.
# -->
# 
# # Development Synaptic remodelling is complicated.
# 
# <img src="30minupdate/pruning.webp" 
#      width="600" 
#      height="750" />
# </h1>
# 
# 
# * Spine density (number of possible post synaptic connections).
# 
# * An initial period when inhibition dominates excitation.
# 
# * Extinction synaptic pruning and neuronal death.
# 
# <!--
# If we don't optimize
# 
# <h1 align="center">
# <img src="30minupdate/Timeline-of-brain-development.png" 
#      width="400" 
#      height="550" />
# </h1>
# 
# 
# <h1 align="center">
# <img src="30minupdate/pruning3.jpg" 
#      width="400" 
#      height="550" />
# </h1>
# -->

# 
# 
# <h1 align="center">
#     What can we use a semi realistic large scale model for?
# </h1>
#     
# * semi realistic, a model that is unrealistic in many respects but is realistic in its scale and its synaptic learning? 
# 
# * Rather than prune, should I just directly change the ratio $ \frac{E}{I} $ from $80%$ E and $20%$ I and the density of E and I synapses.
# 
# 
# 
# 

# * Optimization? 
# 
# * Synaptic Pruning in Neuronal Development (is trajectory and evolution of pruning reproducible)?
#     - Getting from over connected to normally connected (modified STPD).
#     
# * Optimize to find network parameters (synaptic density, neuron parameters) that stop network weight values quickly obtaining maximal values.     
#     
# * Recapitulate Pablos work with predictive coding?
# 
# * Can lateral inhinition cause pruning effect?
# 
# 

# In[10]:



from lava.lib.dnf.operations.operations import Weights
from lava.lib.dnf.operations.operations import *
from lava.proc.lif.process import LIF
from lava.lib.dnf.inputs.rate_code_spike_gen.process import RateCodeSpikeGen
from lava.lib.dnf.connect.connect import connect
from lava.lib.dnf.operations.operations import Weights
from lava.magma.core.run_configs import Loihi1SimCfg #Loihi simulator, not  Loihi itself.
from lava.magma.core.run_conditions import RunSteps
from lava.proc.monitor.process import Monitor
from lava.proc.monitor.models import PyMonitorModel
from lava.lib.dnf.inputs.gauss_pattern.process import GaussPattern
from lava.lib.dnf.kernels.kernels import MultiPeakKernel
from lava.lib.dnf.utils.plotting import raster_plot
from lava.lib.dnf.kernels.kernels import SelectiveKernel


from tutorial06_hierarchical_processes import *
import pandas as pd
import network_params_pynn

import numpy as np
import elephant
get_ipython().system('export JUPYTER_PATH="${JUPYTER_PATH}:$(pwd)/src"')


# In[11]:


K_ext = {
  'L23': {'E': 1600, 'I': 1500},
  'L4' : {'E': 2100, 'I': 1900},
  'L5' : {'E': 2000, 'I': 1900},
  'L6' : {'E': 2900, 'I': 2100}
}

ncolumns=2
# Excitatory

ly_2_3_e = np.ndarray((ncolumns),dtype=object)
ly_4_e = np.ndarray((ncolumns),dtype=object)
ly_5_e = np.ndarray((ncolumns),dtype=object)
ly_6_e = np.ndarray((ncolumns),dtype=object)

# Inhibitory
ly_2_3_i = np.ndarray((ncolumns),dtype=object)
ly_4_i = np.ndarray((ncolumns),dtype=object)
ly_5_i = np.ndarray((ncolumns),dtype=object)
ly_6_i = np.ndarray((ncolumns),dtype=object)


for i in range(0,ncolumns):
    ly_2_3_e[i] = LIF(shape=(K_ext["L23"]["E"],))
    ly_4_e[i] = LIF(shape=(K_ext["L4"]["E"],))
    ly_5_e[i] = LIF(shape=(K_ext["L5"]["E"],))
    ly_6_e[i] = LIF(shape=(K_ext["L6"]["E"],))


    ly_2_3_i[i] = LIF(shape=(K_ext["L23"]["I"],))
    ly_4_i[i] = LIF(shape=(K_ext["L4"]["I"],))
    ly_5_i[i] = LIF(shape=(K_ext["L5"]["I"],))
    ly_6_i[i] = LIF(shape=(K_ext["L6"]["I"],))


# In[12]:


ncells=10
weights = np.random.rand(K_ext["L23"]["E"],K_ext["L4"]["E"])
conn_probs = network_params_pynn.conn_probs

weights_ind = np.where(weights>conn_probs[0][2])
weights[weights_ind] = 0.0

dim=(K_ext["L23"]["E"],K_ext["L23"]["E"])

layer0 = DenseLayer(shape=dim,weights=weights, bias=4, vth=10)
dim=(K_ext["L23"]["E"],K_ext["L23"]["E"])

layer1 = DenseLayer(shape=dim,weights=weights, bias=4, vth=10)
layer0.s_out.connect(layer1.s_in)


# In[ ]:




