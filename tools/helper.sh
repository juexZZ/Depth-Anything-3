#! /bin/bash
# cd da3_streaming/outputs/
# echo "1"
# tar -czvf 12thfloor.tar.gz 12thfloor/
# echo "2"
# tar -czvf 12thfloor_trav1.tar.gz 12thfloor_trav1/
# echo "3"
# tar -czvf 2ndfloor.tar.gz 2ndfloor/
# echo "4"
# tar -czvf 2ndfloor_trav1.tar.gz 2ndfloor_trav1/
# echo "5"
# tar -czvf 28_metfloor.tar.gz 28_metfloor/
# echo "6"
# tar -czvf 28_metfloor_trav1.tar.gz 28_metfloor_trav1/
# echo "7"
# tar -czvf 28_metfloor_trav12.tar.gz 28_metfloor_trav12/

# rsync -avzP -e ssh da3_pcd  jz4725@dtn012.hpc.nyu.edu:/scratch/jz4725/DejaView/pcd_results/

cd /local_data/jz4725/VGGT-SLAM/results/
# tar -czvf 12thfloor.tar.gz 12thfloor/
# tar -czvf 12thfloor_trav1.tar.gz 12thfloor_trav1/
# tar -czvf 2ndfloor.tar.gz 2ndfloor/
# tar -czvf 2ndfloor_trav1.tar.gz 2ndfloor_trav1/
# tar -czvf 28_metfloor.tar.gz 28_metfloor/
tar -czvf 28_metfloor_trav_1.tar.gz 28_metfloor_trav_1/
tar -czvf 28_metfloor_trav_12.tar.gz 28_metfloor_trav_12/

# rsync -avzP -e ssh vggt_pcd  jz4725@dtn012.hpc.nyu.edu:/scratch/jz4725/DejaView/vggt_results/
