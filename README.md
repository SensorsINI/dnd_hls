# Within-Camera Multilayer Perceptron DVS Denoising
Presented at [CVPR 2023 Workshop on Event-based Vision](https://tub-rip.github.io/eventvision2023/)
This is hardware logic code for the event-based MLP denoising filter.

<img src="asic_mlp_dnd/asic_flow/pnr/shot.png" width=400>

This block implements the DVS interface and mulitlayer perceptron of the denoiser in 65nm process technology. Units are microns.

## Abstract
In-camera event denoising can dramatically reduce the data rate of event cameras by filtering out noise at the source. A lightweight multilayer perceptron denoising filter (MLPF) providing state-of-the-art low-cost denoising accuracy processes a small neighborhood of pixels from the timestamp image around each event to discriminate signal and noise events. This paper proposes two digital logic implementations of the MLPF denoiser and quantifies their resource cost,  power, and latency. The hardware MLPF quantizes the weights and hidden unit activations to 4 bits and has about 1k weights with about 40% sparsity. The Area-Under-Curve Receiver Operating Characteristic accuracy is nearly indistinguishable from that of the floating point network. The MLPF processes each event in 10 clock cycles.  In FPGA, it uses 3.5k flip flops and 11.5k LUTs. Our ASIC implementation in 65nm digital technology for a 346 x 260 pixel camera occupies an area of 4.3mm^2 and consumes 4nJ of energy per event at event rates up to 25MHz. The MLPF can be easily integrated into an event camera using an FPGA or as an ASIC directly on the camera chip or in the same package.
This denoising could dramatically reduce the energy consumed by the communication and host processor and open new areas of always-on event camera application under scavenged and battery power.

## Repository folder organization
See the README.md in these folders for details.
- <ins>asic_mlp_dnd</ins>: MLP Denoiser RTL source code fro CMOS 65nm technology.
- <ins>dnd_hls_src</ins>: C++ HLS source code of the MLP Denoiser for Xilinx target devices.
- <ins>hls4ml_model_generation</ins>: hls4ml source files for the MLP HLS code generaion.
- <ins>qmlpf</ins>: qKeras script for MLP model training, quantizing and testing.

## Citation
See the 2023 CVPR Workshop on Event Based Vision paper [Within-Camera Multilayer Perceptron DVS Denoising](https://tub-rip.github.io/eventvision2023/papers/2023CVPRW_Within-Camera_Multilayer_Perceptron_DVS_Denoising_supp.pdf) and the 2022 T-PAMI paper [Low Cost and Latency Event Camera Background Activity Denoising](http://dx.doi.org/10.1109/TPAMI.2022.3152999) on which this work is based.

This project is the result of a collaboration between five labs: Sensors Group, Inst. of Neuroinformatics, UZH-ETH Zurich (UZH-ETH); Robotic and Tech of Computers group, SCORE lab, ETSI-EPS, Univ. of Seville (USE); Univ. of California San Diego (UCSD); Inst. of Particle Physics and Astrophysics, ETH Zurich (ETH); College of Electronic Engineering, National University of Defense Technology (NUDT)

```
@INPROCEEDINGS{Navaro2023-mlpf-dvs-denoising,
  title           = "{Within-Camera} Multilayer Perceptron {DVS} Denoising",
  booktitle       = "2023 {IEEE/CVF} Conference on Computer Vision and Pattern
                     Recognition Workshops ({CVPRW})",
  author          = "Navaro, Antonio Rios and Guo, Shasha and Gnaneswaran,
                     Abarajithan and Vijayakumar, Keerthivasan and Barranco,
                     Alejandro Linares and Aarrestad, Thea and Kastner, Ryan
                     and Delbruck, Tobi",
  institution     = "IEEE",
  year            =  2023,
  location        = "Vancouver"
}


@ARTICLE{Guo2022-am,
  title    = "Low Cost and Latency Event Camera Background Activity Denoising",
  author   = "Guo, Shasha and Delbruck, Tobi",
  journal  = "IEEE Trans. Pattern Anal. Mach. Intell.",
  volume   = "PP",
  month    =  feb,
  year     =  2022,
  url      = "http://dx.doi.org/10.1109/TPAMI.2022.3152999",
  language = "en",
  issn     = "0162-8828",
  pmid     = "35196224",
  doi      = "10.1109/TPAMI.2022.3152999"
}

```
## License
These designs are licensed under the [CERN Open Hardware License CERN-OHL-W (weakly reciprocal)](https://cern-ohl.web.cern.ch/home). See [LICENSE.txt](LICENSE.txt) file for details.
