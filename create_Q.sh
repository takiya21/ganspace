#/bin/bash
function terminate() {
  exit
}
trap 'terminate' {1,2,3,15}

 python create_Q.py --model=StyleGAN2 --class=egg --use_w --layer=style -m 1000 