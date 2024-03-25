# OFPF-MEF
This is the official code for the paper "OFPF-MEF: An Optical Flow Guided Dynamic Multi-exposure Image Fusion Network with Progressive Frequencies Learning"

## Environment Preparing

```
python 3.6
pytorch 1.7.0
visdom 0.1.8.9
dominate 2.6.0
timm 0.6.12
Pillow 9.4.0
DCNv2 0.1
```
### Build DCNv2
```bash
    cd DCNv2/DCN
    ./make.sh         # build
    python testcpu.py    # run examples and gradient check on cpu
    python testcuda.py   # run examples and gradient check on gpu 
```

### Testing

We provide some example images for testing in `./test_data/`. 

To test in static scenarios:
```
python test.py --dataroot your/static/data/root --mode static
```
To test in dynamic scenarios:
```
python test.py --dataroot your/dynamic/data/root --mode dynamic
```
