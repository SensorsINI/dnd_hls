# Synthesise DND model with hls4ml

## Conda

The Python environment used for the tutorials is specified in the env.yml file. It can be setup like:
```
conda env create -f env.yml
conda activate hls4ml-dnd21
```
## Run the code

We target a Xilinx Zynq UltraScale+ MPSoC. Latency is estimated from csynth (myproject_csynth.rpt) and resource consumption from Vivado synthesis (vivado_synth.rpt)
To run
```
python synthesize -Q #runs quantized model
python synthesize -Q #runs keras model converted to ap_fixed<16,6>
```
