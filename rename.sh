#!/bin/bash
for i in {1..45}
do
    mv "models/a2j_dexycb_1/a2j_1_$i.pth" "models/a2j_dexycb_1/a2j_$i.pth"
done