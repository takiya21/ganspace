#/bin/bash
function terminate() {
  exit
}
trap 'terminate' {1,2,3,15}

 python visualize.py --model=StyleGAN2 --class=egg --use_w --layer=style -c=80 -b=10_000 --video --out_dir test_cmp=5stylegan2_batch64x4_310k-iter_fid=38-30437